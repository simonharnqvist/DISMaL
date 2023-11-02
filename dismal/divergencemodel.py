from iclik import inform_crit
import scipy
import math
import numpy as np
import warnings
from dismal.print_results import make_deme_table, make_epoch_table, make_migration_table, print_output
from dismal.demography import Epoch
from dismal import likelihood
from dismal.markov_matrices import TransitionRateMatrix
from decimal import Decimal
from collections import Counter

class DivergenceModel:

    def __init__(self, model_ref=None):
        """Represent two (potentially) diverging demes.

        Args:
            model_ref (str, optional): Model reference. Defaults to None.
        """
        self.model_ref = model_ref
        self.epochs = []

        self.observed_s1 = None
        self.observed_s2 = None
        self.observed_s3 = None
        self.blocklen = None

        self.observed_s1_tally = None
        self.observed_s2_tally = None
        self.observed_s3_tally = None

        self.n_theta_params = 0
        self.n_t_params = -1  # 2 epochs = 1 t, 3 epochs = 2 ts
        self.n_mig_params = 0

        self.thetas_block = None
        self.thetas_site = None
        self.migration_rates = None
        self.ts_theta_scaled = None
        self.ts_2n = None

        self.deme_ids = []

        self.migration_table = None
        self.deme_table = None
        self.epoch_table = None

        self.n_params = None
        self.negll = None
        self.res = None
        self.inferred_params = None
        self.claic = None

        self.fitted_s1 = None
        self.fitted_s2 = None
        self.fitted_s3 = None

        self.optimisation_sucess = None
        self.optimiser = None

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
        self.n_mig_params = self.n_mig_params + epoch.n_mig_params
        self.deme_ids.append(deme_ids)


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

            theta_iv = 1
            t_iv = 1
            m_iv = 0
        
        thetas_iv = [theta_iv] * self.n_theta_params
        ts_iv = [t_iv] * self.n_t_params
        ms_iv = [m_iv] * self.n_mig_params
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
        migr_bounds = [(m_lb, None)] * self.n_mig_params
        bounds = theta_bounds + t_bounds + migr_bounds

        return bounds


    def _generate_markov_chain(self, param_vals):
        """Generate transition rate matrices for a given DivergenceModel given parameter values"""
        Qs = []

        assert self.n_theta_params > 0, "No theta parameters specfied - have you added epochs?"

        thetas = param_vals[0:self.n_theta_params]
        thetas_iter = iter(thetas)
        ms = param_vals[-self.n_mig_params:]
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

        self.observed_s1 = s1
        self.observed_s2 = s2
        self.observed_s3 = s3

        valid_params = self._validate_params(param_vals)
        if not valid_params:
            if verbose:
                warnings.warn(f"Invalid parameters {param_vals}", UserWarning)
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

        assert len(initial_values) == self.n_theta_params + self.n_t_params + self.n_mig_params

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
                self.optimiser = algo
                break
            elif algo_idx < len(opt_algos)-1:
                print(f"Optimiser {algo} failed; trying {opt_algos[algo_idx+1]}")
            else:
                raise RuntimeError(
                    f"Optimisers {opt_algos} all failed to maximise the likelihood")

        assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"
        self.optimisation_sucess = optimised.success
        
        self.inferred_params = optimised.x
        self.negll = optimised.fun
        self.n_params = self.n_theta_params + self.n_t_params + self.n_mig_params
        self.thetas_block = np.array(self.inferred_params[0:self.n_theta_params])
        
        if self.n_mig_params > 0:
            self.migration_rates = self.inferred_params[-self.n_mig_params:]

        self.thetas_site = self.thetas_block/self.blocklen

        self.ts_theta_scaled = self.inferred_params[self.n_theta_params:(self.n_theta_params+self.n_t_params)]
        self.ts_2n = 2*(self.ts_theta_scaled/self.thetas_block[2])

        self._add_params_to_epochs()

        self.migration_table = make_migration_table(self)
        self.epoch_table = make_epoch_table(self)
        self.deme_table = make_deme_table(self)

        self.claic = self._calculate_claic()

        self.observed_s1_tally = self.tally_s_counts(s1)
        self.observed_s2_tally = self.tally_s_counts(s2)
        self.observed_s3_tally = self.tally_s_counts(s3)

        self.fitted_s1 = self._expected_s(state=1, 
                                            scale_by_observed=True,
                                            observed=self.observed_s1_tally,
                                            cutoff=len(self.observed_s1_tally))
        self.fitted_s2 = self._expected_s(state=2, 
                                            scale_by_observed=True,
                                            observed=self.observed_s2_tally,
                                            cutoff=len(self.observed_s2_tally))
        self.fitted_s3 = self._expected_s(state=3, 
                                            scale_by_observed=True,
                                            observed=self.observed_s3_tally,
                                            cutoff=len(self.observed_s3_tally))


    def fit(self,
            s1, s2, s3, blocklen=None,
            initial_values=None,
            bounds=None,
            optimisation_methods=None,
            verbose=False):
        """Estimate parameter values by maximum likelihood.

        Args:
            s1 (ndarray): Counts of segregating sites per locus in state 1.
            s2 (ndarray): Counts of segregating sites per locus in state 2.
            s3 (ndarray): Counts of segregating sites per locus in state 3.
            blocklen (int): Length of loci/blocks used to obtain s counts.
            initial_values (iterable, optional): Initial values for parameter set. Defaults to None.
            bounds (iterable, optional): Bounds in format (min, max) per parameter. Defaults to None.
            optimisation_methods (iterable, optional): Optimisation algorithms to use. 
                See scipy.minimise for full list. Defaults to None, in which case 
                L-BFGS-B, Nelder-Mead, and Powell are attempted sequentially.
            verbose (bool, optional): Whether to print output to console.

        Returns:
            None
        """

        self.observed_s1 = s1
        self.observed_s2 = s2
        self.observed_s3 = s3
        self.blocklen = blocklen
        
        if initial_values is None:
            initial_values = self._get_initial_values(s1=s1, s2=s2, s3=s3, blocklen=blocklen)
        
        if bounds is None:
            bounds = self._get_bounds()

        self._minimise_neg_log_likelihood(s1, s2, s3,
                                                 initial_values,
                                                 bounds,
                                                 optimisation_methods,
                                                 verbose=False)
        
        if verbose is True:
            print_output(self)

        return mod


    def _add_params_to_epochs(self):
        """Update Epoch objects"""
        thetas_iter = iter(self.thetas_site)
        if self.migration_rates is not None:
            ms_iter = iter(self.migration_rates)
        start_times = [0, self.ts_2n[0], sum(self.ts_2n)]
        end_times = [self.ts_2n[0], sum(self.ts_2n), None]

        updated_epochs = []

        for epoch_idx, epoch in enumerate(self.epochs):
            updated_epoch = Epoch(
                deme_ids=epoch.deme_ids,
                migration=epoch.migration,
                asymmetric_migration=epoch.asymmetric_migration,
                migration_direction=epoch.migration_direction)

            updated_epoch.thetas = [next(thetas_iter) for _ in range(epoch.n_thetas)]
            updated_epoch.start_time = start_times[epoch_idx]
            updated_epoch.end_time = end_times[epoch_idx]
            if updated_epoch.migration is True:
                updated_epoch.migration_rates = [next(ms_iter) for _ in range(epoch.n_mig_params)]

            updated_epochs.append(updated_epoch)

        self.epochs = updated_epochs

    
    def _calculate_claic(self):
        """Calculate Composite likelihood AIC."""
        
        def _logl_wrapper(params):
            return self.neg_log_likelihood(params, self.observed_s1, self.observed_s2, self.observed_s3)
        
        try:
            cl_akaike = inform_crit.claic(_logl_wrapper, self.inferred_params)
        except Exception:
            warnings.warn(f"CLAIC could not be calculated", UserWarning)
            cl_akaike = None
        return cl_akaike


    @staticmethod
    def tally_s_counts(counts):
        s_values = [int(i) for i in Counter(counts).keys()]
        s_counts = [int(i) for i in Counter(counts).values()]
        s_arr = np.zeros(shape=(1, max(s_values)+1))

        for idx, s_val in enumerate(s_values):
            s_arr[0, s_val] = s_counts[idx]

        return s_arr[0]


    def _expected_s(self, state, 
                    scale_by_observed=False, observed=None,
                    cutoff=50):
        expected_probs = np.array([likelihood.pr_s(int(k), state,
                            self.thetas_block,
                            self.ts_theta_scaled, 
                            self.migration_rates) for k in range(cutoff)])
        
        if scale_by_observed is True:
            return expected_probs * sum(observed)
        else:
            return expected_probs
        






                    

        
