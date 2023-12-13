import numpy as np
import scipy
from scipy.special import rel_entr
from dismal.demography import Epoch
from dismal.model_instance import ModelInstance
from dismal.demesrepresentation import DemesRepresentation

class DemographicModel:

    def __init__(self, model_ref=None):
        """Represent two (potentially) diverging demes.

        Args:
            model_ref (str, optional): Model reference. Defaults to None.
        """

        self.model_ref = model_ref
        self.epochs = []
        self.deme_ids = []
        self.n_theta_params = 0
        self.n_epoch_durations = -1 # 2 epochs -> 1 duration; 3 epochs -> 2 durations
        self.n_mig_params = 0

        self.modelinstance = None
        self.negll = None
        self.claic = None
        self.inferred_params = None
        self.optim_object = None
        self.stderr = None

        self.demes_representation = None


    def add_epoch(self,
                  n_demes,
                  migration,
                  deme_ids=None,
                  asymmetric_migration=True,
                  migration_direction=None):
        """Add epoch (period of isolation/gene flow).

        Args:
            deme_ids (iterable): Tuples of deme IDs per epoch, specified backwards in time,
              e.g. [("human", "chimp), ("human", "chimp"), ("ancestor",)]
            migration (bool): Allow migration.
            asymmetric_migration (bool, optional): Allow asymmetric migration. Defaults to True.
            migration_direction (iterable, optional): Tuples specifying migration direction
              per epoch with format (source, destination). Defaults to None.
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
        assert len(self.deme_ids) == len(set(self.deme_ids)), f"Deme IDs must be unique across model; found repeated values in {self.deme_ids} "
        self.n_theta_params = self.n_theta_params + epoch.n_demes
        self.n_epoch_durations = self.n_epoch_durations + 1
        self.n_mig_params = self.n_mig_params + epoch.n_mig_params
        self.n_params = self.n_theta_params + self.n_epoch_durations + self.n_mig_params

    
    def _get_initial_values(self, theta_iv=1, epoch_duration_iv=1, mig_iv=0):
        """Generate list of initial values for parameter estimation"""

        assert len(self.epochs) > 1, "Number of epochs must be at least two; add epochs with add_epochs() method"
        
        thetas_iv = [theta_iv] * self.n_theta_params
        epoch_duration_iv = [epoch_duration_iv] * self.n_epoch_durations
        mig_iv = [mig_iv] * self.n_mig_params
        return thetas_iv + epoch_duration_iv + mig_iv
    

    def _get_bounds(self, theta_bounds=(1e-10, None), 
                    epoch_duration_bounds=(1e-10, None), 
                    mig_bounds=(0, None)):
        """Generate list of (lower, upper) bound tuples for parameter estimation"""
        thetas_bounds = [theta_bounds] * self.n_theta_params
        epoch_durations_bounds = [epoch_duration_bounds] * self.n_epoch_durations
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

    
    def fit_model(self, s1, s2, s3, initial_values=None, bounds=None, optimisers=None):
        """Fit model by minimising negative log-likelihood; return ModelInstance with optimised parameters"""

        if initial_values is None:
            initial_values = self._get_initial_values()
        if bounds is None:
            bounds = self._get_bounds()
        
        if optimisers is None:
            opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
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
                print(f"Optimiser {algo} failed; trying {opt_algos[algo_idx+1]}")
            else:
                raise RuntimeError(
                    f"Optimisers {opt_algos} all failed to maximise the likelihood")

        assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"

        self.modelinstance = ModelInstance(optimised.x, self.epochs)
        self.negll = optimised.fun
        self.claic = self.cl_akaike(optimised)
        self.stderr = self.standard_err(optimised)
        self.inferred_params = optimised.x
        self.optim_object = optimised

        self.epochs = self.modelinstance.epochs

        return self

    
    def cl_akaike(self, optim_obj):
        """Calculate composite likelihood AIC from SciPy optimisation object"""
        if hasattr(optim_obj, "jac") and hasattr(optim_obj, "hess_inv"):
            if optim_obj.jac is not None and optim_obj.hess_inv is not None:
                claic = (2*optim_obj.fun 
                        + np.sum(optim_obj.hess_inv.matvec(optim_obj.jac))) # NB sum equiv to trace(diag)
        else:
            claic = None
        
        return claic
    
    def standard_err(self, optim_obj):
        if hasattr(optim_obj, "hess_inv"):
            stderr = np.sqrt(np.diag(optim_obj.hess_inv.todense()))
        else:
            stderr = None
        
        return stderr


    def demes_format(self, mutation_rate, blocklen):
        """Represent model in Demes format"""
        demes_rep = DemesRepresentation(self, mutation_rate=mutation_rate, blocklen=blocklen)
        self.demes_representation = demes_rep
        return demes_rep


    def demesdraw(self, mutation_rate=None, blocklen=None):
        """Draw Demes model; convenience function for demes_format().drawing"""
        if self.demes_representation is None:
            self.demes_representation = self.demes_format(mutation_rate=mutation_rate, blocklen=blocklen)

        return self.demes_representation.drawing


    def demesgraph(self, mutation_rate=None, blocklen=None):
        """Return Demes graph; convenience function for demes_format().graph"""
        if self.demes_representation is None:
            self.demes_representation = self.demes_format(mutation_rate=mutation_rate, blocklen=blocklen)

        return self.demes_representation.graph
    

    def kldiv_fitted_true(self, true_modinst, s_max=500):
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

        observed_sdists = [self.modelinstance.obs_s1, self.modelinstance.obs_s2, self.modelinstance.obs_s3]
        scaled_observed = [dist/np.sum(dist) for dist in observed_sdists]
        
        fitted_sdists = [self.modelinstance.expected_s1(s_max=len(observed_sdists[0])),
                         self.modelinstance.expected_s2(s_max=len(observed_sdists[1])),
                         self.modelinstance.expected_s3(s_max=len(observed_sdists[2]))]
        
        scaled_fitted = [dist/np.sum(dist) for dist in fitted_sdists]
        
        kldiv = np.sum([rel_entr(fitted, obs) for (fitted, obs) 
                          in list(zip(fitted_sdists, observed_sdists))])
        
        return kldiv
        


