import numpy as np
from dismal.likelihood_matrix import LikelihoodMatrix
import scipy

class DivergenceModel:
    """UNTESTED Single model of divergence between two populations."""

    def __init__(
            self,
            epochs_post_div,
            allow_migration,
            symmetric_migration,
            migration_direction,
            population_ids=None
            ):
        
        assert 1 <= epochs_post_div <= 2, "Currently only 2 (1 post-div) and 3 (2 post-div) epoch models are implemented"
        self.epochs_post_div = epochs_post_div
  
        assert isinstance(allow_migration, bool) or isinstance(allow_migration, list), "allow_migration must be bool or list of bools"
        if isinstance(allow_migration, list):
            assert len(allow_migration) == self.epochs_post_div, "Length of allow_migration must be the same as number of epochs"
            assert [isinstance(allow_mig, bool) for allow_mig in allow_migration], "allow_migration should be list of bools, each corresponding to an epoch"
            self.allow_migration = allow_migration
        else:
            self.allow_migration = [allow_migration] * epochs_post_div
        assert isinstance(self.allow_migration, list), "self.allow_migration is not a list of bools (bug, not user error)"

        assert isinstance(symmetric_migration, bool) or isinstance(symmetric_migration, list), "symmetric_migration must be bool or list of bools"
        if isinstance(symmetric_migration, list):
            assert len(symmetric_migration) == self.epochs_post_div, "Length of symmetric_migration must be the same as number of epochs"
            assert [isinstance(allow_mig, bool) for allow_mig in symmetric_migration], "allow_migration should be list of bools, each corresponding to an epoch"
            self.symmetric_migration = symmetric_migration
        else:
            self.symmetric_migration = [symmetric_migration] * epochs_post_div
        assert isinstance(self.symmetric_migration, list), "self.symmetric_migration is not a list of bools (bug, not user error)"

        if isinstance(migration_direction, list):
            assert len(migration_direction) == self.epochs_post_div, "length of migration_direction list must correspond to number of epochs"
            for epoch_idx in range(self.epochs_post_div):
                if migration_direction[epoch_idx] is not None:
                    assert isinstance(migration_direction[epoch_idx], tuple), "migration_direction list must be list of tuples"
                    assert len(migration_direction[epoch_idx]) == 2, "each tuple in list migration_direction must be of len 2 and of format (source, target)"
            self.migration_direction = migration_direction
        else:
            assert migration_direction is None, "migration_direction must be a list of tuples or None"
            self.migration_direction = [None] * self.epochs_post_div
        assert isinstance(self.migration_direction, list) and len(self.migration_direction) == epochs_post_div, "self.migration_direction has wrong type or length (bug, not user error)"

        for epoch_idx in range(self.epochs_post_div):
            if self.allow_migration[epoch_idx] is False:
                assert self.symmetric_migration is False and self.migration_direction == None, f"Migration disallowed in epoch {epoch_idx+1}, but symmetric_migration is True or migration_direction not None"
            elif self.symmetric_migration[epoch_idx] is True:
                assert self.migration_direction[epoch_idx] is None, f"Only symmetric migration allowed in epoch {epoch_idx+1}, but migration direction provided"

        if population_ids is None:
            self.population_ids = [("AB"), ("A1", "B1")]
            if self.epochs_post_div == 2:
                self.population_ids.append(("A2", "B2"))
        else:
            self.population_ids = population_ids
        assert [isinstance(pop, tuple) for pop in self.population_ids], "Population IDs must be given as tuples per epoch"

        self.theta_params = self.get_params()[0]
        self.tau_params = self.get_params()[1]
        self.M_params = self.get_params()[2]
        self.M_tuples = self.migration_rate_tuples()

        self.model_params = self.theta_params + self.tau_params + self.M_params

    def __str__(self):
        return f"""
        DISMaL DivergenceModel object with {len(self.population_ids[-1])} demes and {self.epochs_post_div} epochs post-divergence.
        Populations (per epoch): {self.population_ids}
        Theta (population size) parameters: {self.theta_params}
        Tau (split time) parameters: {self.tau_params}
        M (migration rate) parameters: {self.M_params}
        """

    def __repr__(self):
        return f"""
        DISMaL DivergenceModel object with {len(self.population_ids[-1])} demes and {self.epochs_post_div} epochs post-divergence.
        Populations (per epoch): {self.population_ids}
        Theta (population size) parameters: {self.theta_params}
        Tau (split time) parameters: {self.tau_params}
        M (migration rate) parameters: {self.M_params}
        """
    
    def get_params(self):
        """UNTESTED"""
        taus = [f"tau{i}" for i in range(self.epochs_post_div)]
        assert 0 < len(taus) < 3, "Only 1-2 tau parameters implemented"

        thetas = [("theta_AB", ), ("theta_A1", "theta_B1")]
        if self.epochs_post_div > 1:
            for epoch_idx in range(2, self.epochs_post_div+1):
                thetas.append((f"theta_A{epoch_idx}", f"theta_B{epoch_idx}"))
            
        Ms = []
        for epoch_idx in range(self.epochs_post_div):
            if self.allow_migration[epoch_idx] is False:
                continue
            elif self.symmetric_migration[epoch_idx] is True:
                Ms.append((f"M{epoch_idx+1}", ))
            elif self.migration_direction[epoch_idx] is not None:
                assert len(self.migration_direction[epoch_idx]) == 2
                source = self.migration_direction[epoch_idx][0]
                target = self.migration_direction[epoch_idx][1]
                Ms.append((f"M_{source}_{target}", ))
            else:
                assert self.migration_direction[epoch_idx] is None
                assert self.symmetric_migration[epoch_idx] is False
                pop1, pop2 = self.population_ids[epoch_idx+1]
                Ms.append([(f"M_{pop1}_{pop2}", f"M_{pop2}_{pop1}")])

        return (thetas, taus, Ms)
    
    def migration_rate_tuples(self):
        """UNTESTED"""
        
        migration_rates = []

        for epoch_idx in range(0, self.epochs_post_div):
            pop1, pop2 = self.population_ids[epoch_idx+1][0], self.population_ids[epoch_idx+1][1]
    
            if not self.allow_migration[epoch_idx]:
                mig_rates = False
            elif self.symmetric_migration[epoch_idx] is True:
                mig_rates = True
            elif self.migration_direction[epoch_idx] is not None:
                if self.migration_direction[epoch_idx][0] == pop1:
                    mig_rates = (True, False)
                elif self.migration_direction[epoch_idx][0] == pop2:
                    mig_rates = (False, True)
            else:
                mig_rates = (True, True)

            migration_rates.append(mig_rates)

        return migration_rates

        
    def fit(self, S, initial_values, lower_bounds, verbose=False):
        """UNTESTED"""
        assert isinstance(S, np.nadarray), "S must be np.ndarray"
        assert S.shape[0] == 3, f"S has wrong shape {S.shape}; there should be 3 sampling states"
        assert S.shape[1] > 0, "S contains no data"
            
        if initial_values is not None:
            assert len(initial_values) == 3, "Initial values array should have 3 values: [theta, tau, M]"
        if lower_bounds is not None:
            assert len(lower_bounds) == 3, "Lower bounds array should have 3 values: [theta, tau, M]"

        if initial_values is not None:
            theta_iv = initial_values[0]
            tau_iv = initial_values[1]
            M_iv = initial_values[2]
        else:
            theta_iv = self.estimate_theta()
            tau_iv = self.estimate_tau()
            M_iv = 0

        if lower_bounds is not None:
            theta_lb = initial_values[0]
            tau_lb = initial_values[1]
            M_lb = initial_values[2]
        else:
            theta_lb = 0.01
            tau_lb = 0.01
            M_lb = 0

        thetas_iv = [theta_iv] * len(self.theta_params)
        taus_iv = [tau_iv] * len(self.tau_params)
        Ms_iv = [M_iv] * len(self.M_params) 
        initial_values = thetas_iv + taus_iv + Ms_iv

        thetas_lb = [theta_lb] * len(self.theta_params)
        taus_lb = [tau_lb] * len(self.tau_params)
        Ms_lb = [M_lb] * len(self.M_params)
        theta_bounds = [(thetas_lb, None)] * len(self.theta_params)
        tau_bounds = [(taus_lb, None)] * len(self.tau_params)
        migr_bounds = [(Ms_lb, None)] * len(self.M_params)
        bounds = theta_bounds + tau_bounds + migr_bounds

        return self.minimise_neg_log_likelihood(S, initial_values, bounds, verbose=verbose)
    
    def estimate_theta(self):
        raise NotImplementedError
    
    def estimate_tau(self):
        raise NotImplementedError
        
    def fit_resampled_ivs(self):
        raise NotImplementedError("Fitting with sampled IVs not yet implemented.")
    
    def convert_to_gim_params(self):
        gim_params = ["theta0", "theta1", "theta2", "theta1_prime", "theta2_prime",
                      "t0", "t1", "v", "M1_star", "M2_star", "M1_star_prime", "M2_star_prime"]
        
        n_thetas = len([theta for epoch in self.theta_params for theta in epoch])
        thetas = ["theta0", "theta1", "theta2", "theta1_prime", "theta2_prime"][0:n_thetas]
        assert len(thetas) == (1 + self.epochs_post_div*2), "Number of thetas inconsistent with self.epochs_post_div"
        assert len(thetas)%2 == 1, "Number of thetas should be odd"
        
        if self.epochs_post_div == 1:
            taus = ["tau0"]
        elif self.epochs_post_div == 2:
            taus = ["tau1", "v"]
        else:
            raise ValueError
        
        Ms = []
        param_prefixes = ["M_star", "M_prime_star"]
        for epoch_idx, mig_rates in enumerate(self.M_tuples):
            if len(mig_rates) == 1:
                Ms.append(param_prefixes[epoch_idx])
            elif len(mig_rates) == 2:
                Ms.extend([f"{param_prefixes[epoch_idx]}{n}" for n in [1,2] if mig_rates[n-1] is True])


        return (thetas, taus, Ms)


    @staticmethod
    def composite_likelihood(param_vals, param_names, S, verbose=False):
        """UNTESTED"""
        params = dict(zip(param_names, param_vals))
        likelihood_matrix = LikelihoodMatrix(params, S)
        negll = np.sum(S * likelihood_matrix.matrix)

        if verbose:
            print(params, negll)

        return negll
        
    def minimise_neg_log_likelihood(self, S, initial_values, bounds, verbose):
        """UNTESTED"""
        opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
        for algo_idx, algo in enumerate(opt_algos):
            optimised = scipy.optimize.minimize(self.composite_likelihood,
                                                x0=initial_values,
                                                method=algo,
                                                args=(self.model_params,
                                                      S, verbose),
                                                # options={
                                                #     "ftol": ftol,
                                                #     "eps": eps,
                                                #     "maxfun": maxfun,
                                                #     "maxIs": max_iter
                                                # },
                                                bounds=bounds)
                
            if optimised.success:
                break
            elif algo_idx < len(opt_algos)-1:
                print(f"Optimiser {algo} failed; trying {opt_algos[algo_idx+1]}")
            else:
                raise RuntimeError(
                    f"Optimisers {opt_algos} all failed to maximise the likelihood")

        assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"
        inferred_params = dict(zip(self.model_params, optimised.x))
        n_params = len(self.model_params)
        negll = optimised.fun

        return inferred_params, negll, n_params







