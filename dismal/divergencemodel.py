from dismal.demography import Epoch
from dismal import likelihood, popgen_stats
from dismal.markov_matrices import TransitionRateMatrix
from iclik import info_crit
import scipy
import math
import numpy as np

class DivergenceModel:

    def __init__(self, model_ref=None):
        """Represent two (potentially) diverging demes.

        Args:
            model_ref (str, optional): Model reference. Defaults to None.
        """
        self.model_ref = model_ref
        self.epochs = []

        self.s1 = None
        self.s2 = None
        self.s3 = None

        self.n_theta_params = 0
        self.n_t_params = -1  # 2 epochs = 1 t, 3 epochs = 2 ts
        self.n_m_params = 0

        self.inferred_thetas = None
        self.inferred_ts = None
        self.inferred_ms = None
        self.n_params = None
        self.negll = None
        self.res = None
        self.inferred_params = None
        self.claic = None

    def add_epoch(self,
                  deme_ids,
                  migration,
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
            deme_ids=deme_ids,
            migration=migration,
            asymmetric_migration=asymmetric_migration,
            migration_direction=migration_direction
        )

        self.epochs.append(epoch)
        self.n_theta_params = self.n_theta_params + len(deme_ids)
        self.n_t_params = self.n_t_params + 1
        self.n_m_params = self.n_m_params + epoch.n_m_params


    def _get_initial_values(self, initial_values=None,
                             s1=None, s2=None, s3=None, blocklen=None):
        """Get initial values for theta, t, and m""" 

        if initial_values is not None:
            assert len(initial_values) == 3, "Initial values array should have 3 values: [theta, t, M]"
            theta_iv = initial_values[0]
            t_iv = initial_values[1]
            m_iv = initial_values[2]
        else:
            for s_arr in [s1, s2, s3]:
                assert s_arr is not None, "s1, s2, and s3 required to estimate initial values from data"
            assert isinstance(blocklen, int), "blocklen (int) must be provided if initial values are to be estimated from data" 

            theta_iv = popgen_stats.estimate_pi(s1, s2)
            t_iv = popgen_stats.estimate_dxy(s3, blocklen)/len(self.epochs)
            m_iv = 0
        
        thetas_iv = [theta_iv] * self.n_theta_params
        ts_iv = [t_iv] * self.n_t_params
        ms_iv = [m_iv] * self.n_m_params
        initial_values = thetas_iv + ts_iv + ms_iv

        return initial_values


    def _get_bounds(self, lower_bounds=None):
        """Format lower bounds for optimisation."""
        if lower_bounds is not None:
            assert len(lower_bounds) == 3, "Lower bounds array should have 3 values: [theta, t, m]"
            theta_lb = lower_bounds[0]
            t_lb = lower_bounds[1]
            m_lb = lower_bounds[2]
        else:
            theta_lb = 0.01
            t_lb = 0.01
            m_lb = 0

        theta_bounds = [(theta_lb, None)] * self.n_theta_params
        t_bounds = [(t_lb, None)] * self.n_t_params
        migr_bounds = [(m_lb, None)] * self.n_m_params
        bounds = theta_bounds + t_bounds + migr_bounds

        return bounds


    def _generate_markov_chain(self, param_vals):
        """Generate transition rate matrices for a given DivergenceModel given parameter values"""
        Qs = []

        assert self.n_theta_params > 0, "No theta parameters specfied - have you added epochs?"

        thetas = param_vals[0:self.n_theta_params]
        thetas_iter = iter(thetas)
        ms = param_vals[-self.n_m_params:]
        ms_iter = iter(ms)

        for epoch in self.epochs[:-1]:
            n_thetas_epoch = len(epoch.deme_ids)
            epoch_thetas = [next(thetas_iter) for _ in range(n_thetas_epoch)]

            if epoch.migration is False:
                epoch_ms = [0, 0]
            elif epoch.asymmetric_migration is False:
                epoch_ms = [next(ms_iter)]
            elif epoch.migration_direction is not None:
                assert epoch.migration_direction[0] in epoch.deme_ids and epoch.migration_direction[1] in epoch.deme_ids

                if epoch.migration_direction[0] == epoch.deme_ids[0]:
                    epoch_ms = [next(ms_iter), 0]
                elif epoch.migration_direction[0] == epoch.deme_ids[1]:
                    epoch_ms = [0, next(ms_iter)]
                else:
                    raise ValueError(f"Migration direction {epoch.migration_direction} not compatible with demes in epoch: {epoch.deme_ids}")
            else:
                epoch_ms = [next(ms_iter), next(ms_iter)]

            Q = TransitionRateMatrix(single_deme=False,
                                     thetas=epoch_thetas, 
                                     ms=epoch_ms,
                                     asymmetric_migration=epoch.asymmetric_migration)
            
            Qs.append(Q)

        Qs.append(TransitionRateMatrix(single_deme=True, thetas=[thetas[-1]], ms=[0]))

        return Qs


    @staticmethod
    def _validate_params(param_vals):
        """Check that parameter values are positive and finite."""
        all_positive = all([param >= 0 for param in param_vals])
        contains_nan = any([math.isnan(param) for param in param_vals])
        contains_inf = any([math.isinf(param) for param in param_vals])
    
        return all_positive and not contains_nan and not contains_inf
    
    def neg_log_likelihood(self, param_vals, s1, s2, s3, verbose=False):
        """Calculate negative log-likelihood of parameter set.

        Args:
            param_vals (iterable): Parameter values in order (thetas, ts, ms),
              internally ordered from most recent to least recent (i.e. backwards in time).
            s1 (ndarray): Counts of segregating sites per locus in state 1.
            s2 (ndarray): Counts of segregating sites per locus in state 2.
            s3 (ndarray): Counts of segregating sites per locus in state 3.
            verbose (bool, optional): Whether to print output. Defaults to False.

        Returns:
            float: Negative log-likelihood.
        """

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

        valid_params = self._validate_params(param_vals)
        if not valid_params:
            if verbose:
                print(f"Warning: invalid parameters {param_vals}")
            return np.nan
        else:
            Qs = self._generate_markov_chain(param_vals)
    
            ts = param_vals[self.n_theta_params:(self.n_theta_params+self.n_t_params)]
            assert len(ts) == len(self.epochs)-1, f"Incorrect length {len(ts)} of list ts"

            logl = likelihood.neg_logl(Qs, ts, s1, s2, s3)

            if verbose:
                print(param_vals, logl)
        
            return logl
    
    
    def _minimise_neg_log_likelihood(self,
                                      s1, s2, s3,
                                        initial_values,
                                          bounds,
                                          optimisation_method=None,
                                            verbose=False):
        """Obtain maximum likelihood estimate of parameters by optimisation."""

        assert len(initial_values) == self.n_theta_params + self.n_t_params + self.n_m_params

        if optimisation_method is None:
            opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
        else:
            opt_algos = list(optimisation_method)
        
        for algo_idx, algo in enumerate(opt_algos):
            optimised = scipy.optimize.minimize(self.neg_log_likelihood,
                                                x0=initial_values,
                                                method=algo,
                                                args=(s1, s2, s3, verbose),
                                                bounds=bounds)
            if optimised.success:
                break
            elif algo_idx < len(opt_algos)-1:
                print(f"Optimiser {algo} failed; trying {opt_algos[algo_idx+1]}")
            else:
                raise RuntimeError(
                    f"Optimisers {opt_algos} all failed to maximise the likelihood")

        assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"
        
        self.inferred_params = optimised.x
        self.negll = -optimised.fun
        self.n_params = self.n_theta_params + self.n_t_params + self.n_m_params
        self.inferred_thetas = self.inferred_params[0:self.n_theta_params]
        self.inferred_ts = self.inferred_params[self.n_theta_params:(self.n_theta_params+self.n_t_params)]
        if self.n_m_params > 0:
            self.inferred_ms = self.inferred_params[-self.n_m_params:]
        self.claic = self.calculate_claic()
        self.res = self._results_dict()


    def fit(self,
            s1, s2, s3, blocklen=None,
            initial_values = None,
            bounds = None,
            optimisation_methods=None):
        """Estimate parameter values by maximum likelihood.

        Args:
            s1 (ndarray): Counts of segregating sites per locus in state 1.
            s2 (ndarray): Counts of segregating sites per locus in state 2.
            s3 (ndarray): Counts of segregating sites per locus in state 3.
            blocklen (int, optional): Length of loci/blocks used to obtain s counts.
              Required unless initial values are provided. Defaults to None.
            initial_values (iterable, optional): Initial values for parameter set. Defaults to None.
            bounds (iterable, optional): Bounds in format (min, max) per parameter. Defaults to None.
            optimisation_methods (iterable, optional): Optimisation algorithms to use. 
                See scipy.minimise for full list. Defaults to None, in which case 
                L-BFGS-B, Nelder-Mead, and Powell are attempted sequentially.

        Returns:
            dict: Dictionary of results of optimisation.
        """

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        
        if initial_values is None:
            initial_values = self._get_initial_values(s1=s1, s2=s2, s3=s3, blocklen=blocklen)
        
        if bounds is None:
            bounds = self._get_bounds()

        return self._minimise_neg_log_likelihood(s1, s2, s3,
                                                 initial_values,
                                                 bounds,
                                                 optimisation_methods,
                                                 verbose=False)
    
    def _results_dict(self):
        return {
            "model_ref": self.model_ref,
            "n_epochs": len(self.epochs),
            "demes": [[i for i in epoch.deme_ids] for epoch in self.epochs],
            "thetas": self.inferred_thetas,
            "ts": self.inferred_ts,
            "mig_rates": self.inferred_ms,
            "n_params": self.n_params,
            "neg_log_likelihood": self.negll,
            "claic": self.claic
        }
    
    @staticmethod
    def from_dict_spec(model_spec):
        """Generate DivergenceModel object from dictionary specification.

        Args:
            model_spec (dict): Dictionary with keys
              "deme_ids", "model_ref", "epochs", "migration", "migration_direction", "asym_migration".
              Example: {"model_ref": "gim_symmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (True, True, False), "asym_migration": (False, False, False)}])

        Returns:
            DivergenceModel: Model object with specified settings.
        """
        deme_ids = model_spec["deme_ids"]

        if "model_ref" in model_spec.keys():
            model_ref = model_spec["model_ref"]
        else:
            model_ref = None

        mod = DivergenceModel(model_ref=model_ref)

        for epoch_idx in range(model_spec["epochs"]):
            allow_migration_epoch = model_spec["migration"][epoch_idx]
            assert isinstance(allow_migration_epoch, bool)

            if allow_migration_epoch:

                if "asym_migration" in model_spec.keys():
                    allow_asymmetric_migration_epoch = model_spec["asym_migration"][epoch_idx]
                else:
                    allow_asymmetric_migration_epoch = False

                if "migration_direction" in model_spec.keys():
                    migration_direction_epoch = model_spec["migration_direction"][epoch_idx]
                    assert len(migration_direction_epoch) == 2, "Migration direction must be specified as (source, target) tuple per epoch"

                    assert [migration_direction_epoch[deme_idx] in deme_ids[epoch_idx] for deme_idx in [0,1]]
                else:
                    migration_direction_epoch = None

            else:
                allow_asymmetric_migration_epoch = False
                migration_direction_epoch = None

            mod.add_epoch(deme_ids = deme_ids[epoch_idx],
                            migration = allow_migration_epoch,
                            asymmetric_migration = allow_asymmetric_migration_epoch,
                            migration_direction = migration_direction_epoch)

        return mod
    
    def _calculate_claic(self):
        """Calculate Composite likelihood AIC."""
        
        def _logl_wrapper(params):
            return self.neg_log_likelihood(params, self.s1, self.s2, self.s3)
        
        cl_akaike = info_crit.claic(_logl_wrapper, self.inferred_params)
        self.claic = cl_akaike

        

        
        



