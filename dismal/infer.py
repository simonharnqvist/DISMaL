import numpy as np

class DivergenceModel:
    """UNTESTED Single model of divergence between two populations."""

    def __init__(
            self,
            allow_migration,
            epochs_post_div,
            symmetric_migration,
            migration_direction,
            population_ids=None
    ):
        

        assert isinstance(allow_migration, list), "allow_migration should be list of bools, each corresponding to an epoch  e.g. [True, True]"
        assert [isinstance(allow_migration[i], bool) for i in allow_migration], "allow_migration should be list of bools, each corresponding to an epoch  e.g. [True, True]"
        assert 1 <= epochs_post_div <= 2, "Currently only 2 (1 post-div) and 3 (2 post-div) epoch models are implemented"
        
        if symmetric_migration is not None:
            assert isinstance(symmetric_migration, list), "symmetric_migration should be list of bools, each corresponding to an epoch  e.g. [True, True]"
            assert [isinstance(symmetric_migration[i], bool) for i in symmetric_migration], "symmetric_migration should be list of bools, each corresponding to an epoch  e.g. [True, True]"
        
        if migration_direction is not None:
            assert isinstance(migration_direction, list), "migration_direction should be list of bools, each corresponding to an epoch  e.g. [True, True]"
            assert [isinstance(migration_direction[i], bool) for i in migration_direction], "unidirectional_migration should be list of bools, each corresponding to an epoch  e.g. [True, True]"

        self.epochs_post_div = epochs_post_div
        self.allow_migration = allow_migration
        self.symmetric_migration = symmetric_migration
        self.migration_direction = migration_direction

        # unidirectional implies asymmetric
        for m in range(len(self.migration_direction)):
                if self.migration_direction is not None:
                    self.symmetric_migration[m] = True

        if population_ids is None:
            self.population_ids = [("AB"), ("A1", "B1")]
            if self.epochs_post_div == 2:
                self.population_ids.append(("A2", "B2"))
        assert [isinstance(tuple, population_ids[i]) for i in range(len(population_ids))], "Population IDs must be given as tuples per epoch"

        self.theta_params = self.get_params()[0]
        self.tau_params = self.get_params()[1]
        self.M_params = self.get_params()[2]


        def __str__(self):
            return f"""
            DISMaL DivergenceModel object.
            Theta (population size) parameters: {self.thetas}
            Tau (split time) parameters: {self.taus}
            M (migration rate) parameters: {self.Ms}
            """

        def __repr__(self):
            return f"""
            DISMaL DivergenceModel object.
            Theta (population size) parameters: {self.thetas}
            Tau (split time) parameters: {self.taus}
            M (migration rate) parameters: {self.Ms}
            """
        
        def get_params(self):
            """UNTESTED"""
            taus = [f"tau{i}" for i in range(self.epochs_post_div)]
            assert 0 < len(taus) < 3, "Only 1-2 tau parameters implemented"

            thetas = ["thetaAB", "thetaA1", "thetaB1"] 
            if self.epochs_post_div > 1:
                thetas = thetas + ["thetaA{n}", "thetaB{n}" for n in range(2, self.epochs_post_div)]
            
            Ms = []
            for epoch_idx in range(self.epochs_post_div):
                if self.symmetric_migration[epoch_idx] is True:
                    Ms.append(f"M{epoch_idx}")
                elif self.migration_direction[epoch_idx] is not None:
                    assert len(self.migration_direction[epoch_idx]) == 2
                    source = migration_direction[epoch_idx][0]
                    target = migration_direction[epoch_idx][1]
                    Ms.append(f"M_{source}_{target}")
                else:
                    assert self.migration_direction[epoch_idx] is None
                    assert self.symmetric_migration[epoch_idx] is False

                    pop1, pop2 = population_ids[epoch_idx]
                    Ms.extend([f"M_{pop1}_{pop2}", f"M_{pop2}_{pop1}"])

            return (thetas, taus, Ms)

        
        def fit(S, initial_values, lower_bounds, verbose=False):
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

            thetas_iv = [self.theta_iv] * len(self.theta_params)
            taus_iv = [self.tau_iv] * len(self.tau_params)
            Ms_iv = [self.M_iv] * len(self.M_params) 
            initial_values = thetas_iv + taus_iv + Ms_iv

            thetas_lb = [self.theta_lb] * len(self.theta_params)
            taus_lb = [self.tau_lb] * len(self.tau_params)
            Ms_lb = [self.M_lb] * len(self.M_params)
            theta_bounds = [(thetas_lb, None)] * len(self.theta_params)
            tau_bounds = [(taus_lb, None)] * len(self.tau_params)
            migr_bounds = [(Ms_lb, None)] * len(self.M_params)
            bounds = theta_bounds + tau_bounds + migr_bounds

            return self.minimise_neg_log_likelihood(S, initial_values, bounds, verbose=verbose)
        
        def fit_resampled_ivs(self):
            raise NotImplementedError("Fitting with sampled IVs not yet implemented.")

        @staticmethod
        def composite_likelihood(param_vals, param_names, S, verbose=False):
            """UNTESTED"""
            params = dict(zip(param_names, param_vals))
            likelihood_matrix = LikelihoodMatrix(params, S)
            negll = np.sum(S * likelihood_matrix.matrix)

            if verbose:
                print(params, negll)

            return negll
        
        def minimise_neg_log_likelihood(self, S, initial_values, bounds, verbose=verbose):
            """UNTESTED"""
            opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
            for i in range(len(opt_algos)):
                optimised = scipy.optimize.minimize(self.composite_likelihood,
                                                x0=initial_values,
                                                method=opt_algos[i],
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
                elif i < len(opt_algos)-1:
                    print(f"Optimiser {opt_algos[i]} failed; trying {opt_algos[i+1]}")
                else:
                    raise RuntimeError(
                        f"Optimisers {opt_algos} all failed to maximise the likelihood")

            assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"
            inferred_params = dict(zip(self.model_params, optimised.x))
            n_params = len(self.model_params)
            negll = optimised.fun

            return inferred_params, negll, n_params







