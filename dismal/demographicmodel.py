import numpy as np
import scipy
from scipy.special import rel_entr
from dismal.demography import Epoch
from dismal.model_instance import ModelInstance
from dismal.modelsimulation import ModelSimulation
from dismal.demesrepresentation import DemesRepresentation
import warnings
import tqdm
from copy import deepcopy

np.set_printoptions(suppress=True)

#  TODO: replace with DemographicModel from ABISS


class DemographicModel:

    def __init__(self, model_ref: str = None) -> None:
        """Represent single demographic model.

        :param model_ref: Model name, defaults to None
        :type model_ref: str, optional
        """

        self.model_ref = model_ref
        self.epochs = []
        self.deme_ids = []
        self.n_theta_params = 0
        self.n_epoch_durations = -1  # 2 epochs -> 1 duration; 3 epochs -> 2 durations
        self.n_mig_params = 0

        self.modelinstance = None
        self.negll = None
        self.aic = None
        self.claic = None
        self.inferred_params = None
        self.optim_object = None
        self.stderr = None
        self.bootstraps = None

        self.demes_representation = None

    def add_epoch(self,
                  n_demes: int,
                  migration: bool,
                  deme_ids: list[tuple[str]] = None,
                  asymmetric_migration: bool = True,
                  migration_direction: list[tuple[str]] = None) -> None:
        """Add an epoch

        :param n_demes: Number of demes in epoch
        :type n_demes: int
        :param migration: Whether to allow migration
        :type migration: bool
        :param deme_ids: Deme names, defaults to None
        :type deme_ids: list[tuple[str]], optional
        :param asymmetric_migration: Whether to allow asymmetric migration, defaults to True
        :type asymmetric_migration: bool, optional
        :param migration_direction: Direction of migration e.g. ("A", "B") for backwards-in-time migration A->B, defaults to None
        :type migration_direction: list[tuple[str]], optional
        """

        epoch = Epoch(
            n_demes=n_demes,
            deme_ids=deme_ids,
            migration=migration,
            asymmetric_migration=asymmetric_migration,
            migration_direction=migration_direction
        )

        self.epochs.append(epoch)
        self.deme_ids.append(epoch.deme_ids)
        assert len(self.deme_ids) == len(set(self.deme_ids)
                                         ), f"Deme IDs must be unique across model; found repeated values in {self.deme_ids} "
        self.n_theta_params = self.n_theta_params + epoch.n_demes
        self.n_epoch_durations = self.n_epoch_durations + 1
        self.n_mig_params = self.n_mig_params + epoch.n_mig_params
        self.n_params = self.n_theta_params + self.n_epoch_durations + self.n_mig_params

    def _get_initial_values(self, theta_iv=1, epoch_duration_iv=1, mig_iv=1e-5):
        """Generate list of initial values for parameter estimation"""

        assert len(
            self.epochs) > 1, "Number of epochs must be at least two; add epochs with add_epochs() method"

        thetas_iv = [theta_iv] * self.n_theta_params
        epoch_duration_iv = [epoch_duration_iv] * self.n_epoch_durations
        mig_iv = [mig_iv] * self.n_mig_params
        return thetas_iv + epoch_duration_iv + mig_iv

    def _get_bounds(self, theta_bounds=(1e-10, None),
                    epoch_duration_bounds=(1e-10, None),
                    mig_bounds=(0, None)):
        """Generate list of (lower, upper) bound tuples for parameter estimation"""
        thetas_bounds = [theta_bounds] * self.n_theta_params
        epoch_durations_bounds = [
            epoch_duration_bounds] * self.n_epoch_durations
        migs_bounds = [mig_bounds] * self.n_mig_params
        return thetas_bounds + epoch_durations_bounds + migs_bounds

    def _negll_wrapper(self, parameter_values, s1, s2, s3):
        """Calculate negative composite log-likelihood"""
        try:
            mod = ModelInstance(parameter_values, self.epochs)
            lnl = mod.neg_composite_log_likelihood(s1, s2, s3)
        except scipy.linalg.LinAlgError:
            lnl = np.nan
        except ValueError:
            lnl = np.nan

        return lnl

    def fit_model(self, s1: np.array, s2: np.array, s3: np.array,
                  initial_values: list = None,
                  bounds: list[tuple] = None,
                  optimisers: list = None) -> None:
        """Fit model to estimate parameters

        :param s1: Segregating sites distribution within population 1
        :type s1: np.array
        :param s2: Segregating sites distribution within population 2
        :type s2: np.array
        :param s3: Segregating sites distribution between populations
        :type s3: np.array
        :param initial_values: Initial values for optimisation, defaults to None in which case defaults are used
        :type initial_values: list, optional
        :param bounds: Bounds for parameters (low, high), defaults to None
        :type bounds: list[tuple], optional
        :param optimisers: Optimisation algorithms to use, defaults to None in which case L-BFGS-B and Nelder-Mead are tried sequentially
        :type optimisers: list, optional
        :raises RuntimeError: If all optimisers fail
        """

        if initial_values is None:
            initial_values = self._get_initial_values()
        if bounds is None:
            bounds = self._get_bounds()

        if optimisers is None:
            opt_algos = ["L-BFGS-B", "Nelder-Mead"]
        else:
            opt_algos = list(optimisers)

        for algo_idx, algo in enumerate(opt_algos):
            optimised = scipy.optimize.minimize(self._negll_wrapper,
                                                x0=initial_values,
                                                method=algo,
                                                args=(s1, s2, s3),
                                                bounds=bounds)
            if optimised.success:
                self.optimiser = algo
                break
            elif algo_idx < len(opt_algos)-1:
                print(
                    f"Optimiser {algo} failed; trying {opt_algos[algo_idx+1]}")
            else:
                raise RuntimeError(
                    f"Optimisers {opt_algos} all failed to maximise the likelihood")

        assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"

        self.modelinstance = ModelInstance(optimised.x, self.epochs)
        self.negll = optimised.fun
        self.aic = 2*self.n_params + 2*self.negll
        self.claic = self._cl_akaike(optimised)
        self.stderr = self._standard_err(optimised)
        self.inferred_params = optimised.x
        self.optim_object = optimised

        self.epochs = self.modelinstance.epochs

        # return self

    def _bounds_warnings(self):

        thetas = self.inferred_params[0:self.n_theta_params]
        if any(thetas <= 1e-3):
            warnings.warn(
                "One or more of your theta values is very low. Consider increasing the blocklength. Interpret results with great caution.")
        if any(thetas >= 20):
            warnings.warn(
                "One or more of your theta values is much greater than expected. Consider decreasing the block length. Interpret results with great caution.")

        epoch_durations = self.inferred_params[self.n_theta_params:(
            self.n_theta_params+self.n_epoch_durations)]
        if any(epoch_durations <= 1e-5):
            warnings.warn(
                "One or more of your epoch duration values is very low. Interpret results with great caution.")
        if any(epoch_durations > 20):
            warnings.warn(
                "One or more of your epoch duration values is much greater than expected. Interpret results with great caution.")

        if self.n_mig_params > 0:
            mig_rates = self.inferred_params[(
                self.n_theta_params+self.n_epoch_durations):]

            if any(mig_rates <= 1e-5):
                warnings.warn(
                    "One or more of your migration rates is very low. Interpret results with great caution.")
            if any(mig_rates > 20):
                warnings.warn(
                    "One or more of your migration rates is much greater than expected. Interpret results with great caution.")

    def _cl_akaike(self, optim_obj):
        """Calculate composite likelihood AIC from SciPy optimisation object"""
        # Important note: it does not matter that hes and jac are from negative logl
        # since they are multiplied together, yielding same result as if both were negated first
        if hasattr(optim_obj, "jac") and hasattr(optim_obj, "hess_inv"):
            if optim_obj.jac is not None and optim_obj.hess_inv is not None:
                claic = (2*optim_obj.fun
                         + 2*np.sum(optim_obj.hess_inv.matvec(optim_obj.jac)))  # NB sum equiv to trace(diag)
        else:
            claic = None

        if claic is not None:
            if claic < 0:
                claic = None

        return claic

    def _standard_err(self, optim_obj):
        if hasattr(optim_obj, "hess_inv"):
            stderr = np.sqrt(np.diag(optim_obj.hess_inv.todense()))
        else:
            stderr = None

        return stderr

    def demes_format(self, mutation_rate, blocklen, log_time=True):
        """Represent model in Demes format - to be removed in next refactor"""
        demes_rep = DemesRepresentation(
            self, mutation_rate=mutation_rate, blocklen=blocklen, log_time=log_time)
        self.demes_representation = demes_rep
        return demes_rep

    def demesdraw(self, mutation_rate=None, blocklen=None, log_time=True):
        """Draw Demes model; convenience function for demes_format().drawing - to be removed in next refactor"""
        if self.demes_representation is None:
            self.demes_representation = self.demes_format(
                mutation_rate=mutation_rate, blocklen=blocklen, log_time=log_time)

        return self.demes_representation.drawing

    def kldiv_fitted_true(self, true_modinst: ModelInstance, s_max: int = 500):
        """Evaluate model fit (KL-divergence) against specified parameter set ModelInstance"""

        assert isinstance(true_modinst, ModelInstance)

        true_sdists = [true_modinst.expected_s1(s_max=s_max),
                       true_modinst.expected_s2(s_max=s_max),
                       true_modinst.expected_s3(s_max=s_max)]

        scaled_true = [dist/np.sum(dist) for dist in true_sdists]

        fitted_sdists = [self.modelinstance.expected_s1(s_max=s_max),
                         self.modelinstance.expected_s2(s_max=s_max),
                         self.modelinstance.expected_s3(s_max=s_max)]

        scaled_fitted = [dist/np.sum(dist) for dist in fitted_sdists]

        kldiv = np.sum([rel_entr(fitted, true) for (fitted, true)
                        in list(zip(scaled_fitted, scaled_true))])

        return kldiv

    def kldiv_fitted_observed(self):
        """Evaluate model fit (KL-divergence) against observed data"""

        observed_sdists = [self.modelinstance.obs_s1,
                           self.modelinstance.obs_s2, self.modelinstance.obs_s3]
        scaled_observed = [dist/np.sum(dist) for dist in observed_sdists]

        fitted_sdists = [self.modelinstance.expected_s1(s_max=len(observed_sdists[0])),
                         self.modelinstance.expected_s2(
                             s_max=len(observed_sdists[1])),
                         self.modelinstance.expected_s3(s_max=len(observed_sdists[2]))]

        scaled_fitted = [dist/np.sum(dist) for dist in fitted_sdists]

        kldiv = np.sum([rel_entr(fitted, obs) for (fitted, obs)
                        in list(zip(fitted_sdists, observed_sdists))])

        return kldiv

    def bootstrap_mle(self, mutation_rate: float,
                      blocklen: int,  recombination_rate: float = 0,
                      n_bootstraps: int = 100) -> np.array:
        """_summary_

        :param mutation_rate: Mutation rate
        :type mutation_rate: float
        :param recombination_rate: Recombination rate (note that original model was fitted assuming no recombination)
        :type recombination_rate: float
        :param blocklen: Block length
        :type blocklen: int
        :param n_bootstraps: Number of bootstrap replicates, defaults to 100
        :type n_bootstraps: int, optional
        :return: Bootstrap values
        :rtype: np.array
        """

        res = []
        for _ in tqdm.tqdm(range(n_bootstraps)):
            try:
                sim = ModelSimulation(self.modelinstance, mutation_rate=mutation_rate,
                                      recombination_rate=recombination_rate, blocklen=blocklen)
                new_mod = DemographicModel()
                new_mod.epochs = self.epochs
                new_mod.deme_ids = self.deme_ids
                new_mod.n_epoch_durations = self.n_epoch_durations
                new_mod.n_mig_params = self.n_mig_params
                new_mod.n_theta_params = self.n_theta_params
                new_mod.n_params = self.n_params
                new_mod.fit_model(sim.s1, sim.s2, sim.s3)
                res.append(new_mod.inferred_params)
            except RuntimeError:
                continue

        self.bootstraps = np.array(res)
        return np.array(res)
