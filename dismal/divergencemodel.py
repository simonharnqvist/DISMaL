from dismal.demography import Epoch
from dismal import likelihood, popgen_stats
from dismal.markov_matrices import TransitionRateMatrix
import scipy

class DivergenceModel:

    def __init__(self):
        self.model = None
        self.epochs = []

        self.n_theta_params = 0
        self.n_t_params = -1  # 2 epochs = 1 t, 3 epochs = 2 ts
        self.n_m_params = 0

    def get_parameter_idxs(self):
        theta_param_idxs = list(range(0, self.n_theta_params))
        t_param_idxs = list(range(max(theta_param_idxs)+1, max(theta_param_idxs)+1+self.n_t_params))
        m_param_idxs = list(range(max(t_param_idxs)+1, max(t_param_idxs)+1+self.n_m_params))

        return (theta_param_idxs, t_param_idxs, m_param_idxs)

    def add_epoch(self,
                  index,
                  deme_ids,
                  migration,
                  asymmetric_migration=True,
                  migration_direction=None):
        
        epoch = Epoch(
            index=index,
            deme_ids=deme_ids,
            migration=migration,
            asymmetric_migration=asymmetric_migration,
            migration_direction=migration_direction
        )

        self.epochs.append(epoch)
        self.n_theta_params = self.n_theta_params + len(deme_ids)
        self.n_t_params = self.n_t_params + 1
        self.n_m_params = self.n_m_params + epoch.n_m_params

    def get_initial_values(self, initial_values=None, s1=None, s2=None, s3=None, blocklen=None): 

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
            t_iv = popgen_stats.estimate_dxy(s3, blocklen)
            m_iv = 0
        
        thetas_iv = [theta_iv] * self.n_theta_params
        ts_iv = [t_iv] * self.n_t_params
        ms_iv = [m_iv] * self.n_m_params
        initial_values = thetas_iv + ts_iv + ms_iv

        return initial_values
    
    def get_bounds(self, lower_bounds=None):
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
    
    def generate_markov_chain(self, param_vals):
        """Generate transition rate matrices for a given DivergenceModel given parameter values"""
        Qs = []

        param_idxs = self.get_parameter_idxs()
        
        thetas_iter = iter(param_vals[param_idxs[0]])
        ms_iter = iter(param_vals[param_idxs[2]])

        for epoch in self.epochs[:-1]:
            n_thetas_epoch = len(epoch.deme_ids)
            epoch_thetas=[next(thetas_iter) for _ in range(n_thetas_epoch)]

            n_ms_epoch = epoch.n_m_params
            if n_ms_epoch > 0:
                epoch_ms = [next(ms_iter) for _ in range(n_ms_epoch)]
            else:
                epoch_ms = []

            Q = TransitionRateMatrix(thetas=epoch_thetas, 
                                     ms=epoch_ms,
                                     asymmetric_migration=epoch.asymmetric_migration)
            
            Qs.append(Q)

        Qs.append(TransitionRateMatrix(thetas=[param_vals[param_idxs[0][-1]]], ms=[0])) # 'ancestral epoch' NB wrong - should be changed later for clarity but no impact

        return Qs
    
    def _log_likelihood_from_params(self, param_vals, s1, s2, s3, verbose=False):
        Qs = self.generate_markov_chain(param_vals)
        ts = param_vals[self.get_parameter_idxs()[1]]
        logl = likelihood.log_likelihood(Qs, ts, s1, s2, s3)

        if verbose:
            print(param_vals, logl)
        
        return logl
    
    def minimise_neg_log_likelihood(self, s1, s2, s3, initial_values, bounds, verbose=False):
        opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
        for algo_idx, algo in enumerate(opt_algos):
            optimised = scipy.optimize.minimize(self._log_likelihood_from_params,
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
        inferred_params = optimised.x
        n_params = len(initial_values)  # need to combine these back to make human readable output
        negll = optimised.fun
        aic = 2*n_params + 2*negll

        return inferred_params, negll, n_params, aic


    def fit(self,
            s1, s2, s3, blocklen=None,
            initial_values = None,
            bounds = None):
        
        if initial_values is None:
            initial_values = self.get_initial_values(s1=s1, s2=s2, s3=s3, blocklen=blocklen)
        
        if bounds is None:
            bounds = self.get_bounds()

        return self.minimise_neg_log_likelihood(s1, s2, s3, initial_values, bounds, verbose=False)

        

        
        



