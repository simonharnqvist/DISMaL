import numpy as np
from dismal.likelihood_matrix import LikelihoodMatrix
from dismal import demography, popgen_stats
from dismal.demography import Epoch, Population
import scipy
import pandas as pd

class DivergenceModel:
    """UNTESTED Single model of divergence between two populations."""

    def __init__(
            self,
            epochs,
            allow_migration,
            symmetric_migration=False,
            migration_direction=None,
            population_names=None
            ):
        
        assert 2 <= epochs <= 3, "Number of epochs must be either 2 or 3"
        self.n_epochs = epochs

        self.epochs = []
        self.populations = []

        self.n_populations = self.n_epochs*2 + 1
        if population_names is None:
            self.population_names = [None] * self.n_populations
    
        self.populations.append(Population(id=0, epoch_id=0, population_name=self.population_names[0]))
        self.epochs.append(Epoch(id=0, allow_migration=False, populations=[0]))

        populations = iter(list(zip(range(1, self.n_populations), self.population_names)))

        if isinstance(allow_migration, bool):
            allow_migration = [allow_migration] * (self.n_epochs-1)
        self.allow_migration = allow_migration
        assert isinstance(self.allow_migration, list), "allow_migration must be list of bools of len epochs-1"

        if isinstance(symmetric_migration, bool):
            symmetric_migration = [symmetric_migration] * (self.n_epochs-1)
        self.symmetric_migration = symmetric_migration
        assert isinstance(self.symmetric_migration, list), "symmetric_migration must be list of bools of len epochs-1"

        if isinstance(migration_direction, tuple):
            migration_direction = [migration_direction] * (self.n_epochs-1)
        elif migration_direction is None:
            migration_direction = [None] * (self.n_epochs-1)
        self.migration_direction = migration_direction
        assert isinstance(self.migration_direction, list), "migration_direction must be list of tuples or bools"

        for epoch_idx in range(1, self.n_epochs):
            epoch_pop1, epoch_pop2 = next(populations), next(populations)
            pop1 = Population(id=epoch_pop1[0], epoch_id=epoch_idx, population_name=epoch_pop1[1])
            pop2 = Population(id=epoch_pop2[0], epoch_id=epoch_idx, population_name=epoch_pop2[1])
            self.populations.extend([pop1, pop2])

            assert isinstance(self.allow_migration[epoch_idx-1], bool), f"allow_migration for epoch {epoch_idx} not bool"
            assert isinstance(self.symmetric_migration[epoch_idx-1], bool), f"symmetric_migration for epoch {epoch_idx} not bool"
            if self.migration_direction[epoch_idx-1] is not None:
                assert isinstance(self.migration_direction[epoch_idx-1], tuple), f"symmetric_migration for epoch {epoch_idx} not None or tuple"
                epoch_pops_ids = [pop1.id, pop2.id]
                migration_source = self.migration_direction[epoch_idx-1][0]
                migration_target = self.migration_direction[epoch_idx-1][1]
                assert migration_source in epoch_pops_ids, f"Population {migration_source} not in populations in epoch {epoch_idx}"
                assert migration_target in epoch_pops_ids, f"Population {migration_target} not in populations in epoch {epoch_idx}"

            epoch = Epoch(id=epoch_idx,
                          allow_migration=self.allow_migration[epoch_idx-1],
                          symmetric_migration=self.symmetric_migration[epoch_idx-1],
                          migration_direction=self.migration_direction[epoch_idx-1],
                          populations=[pop1.id, pop2.id])
            
            self.epochs.append(epoch)

        self.population_ids = [population.id for population in self.populations]
        self.theta_params = [f"theta_{id}" for id in self.population_ids]
        self.tau_params = [f"tau_{t}" for t in range(1, self.n_epochs)]

        self.n_theta_params = len(self.population_ids)
        self.n_tau_params = self.n_epochs-1
        self.n_M_params = sum([epoch.n_M_params for epoch in self.epochs])

    def __str__(self):
        return f"""
        DISMaL DivergenceModel object with {self.n_populations} demes and {self.n_epochs} epochs.
        BLAH BLAH more stuff here
        """

    def __repr__(self):
        return f"""
        DISMaL DivergenceModel object with {self.n_populations} demes and {self.n_epochs} epochs.
        BLAH BLAH more stuff here
        """
    
    def get_initial_values(self, S, initial_values=None, blocklen=None): 
        if initial_values is not None:
            assert len(initial_values) == 3, "Initial values array should have 3 values: [theta, tau, M]"

        if initial_values is not None:
            theta_iv = initial_values[0]
            tau_iv = initial_values[1]
            M_iv = initial_values[2]
        else:
            theta_iv = popgen_stats.estimate_pi(S)
            assert isinstance(blocklen, int), "blocklen (int) must be provided if initial values are to be estimated from data" 
            tau_iv = popgen_stats.estimate_dxy(S, blocklen)
            M_iv = 0
        
        thetas_iv = [theta_iv] * self.n_theta_params
        taus_iv = [tau_iv] * self.n_tau_params
        Ms_iv = [M_iv] * self.n_M_params
        initial_values = thetas_iv + taus_iv + Ms_iv

        return initial_values


    def get_bounds(self, S, lower_bounds=None):
        if lower_bounds is not None:
            assert len(lower_bounds) == 3, "Lower bounds array should have 3 values: [theta, tau, M]"

        if lower_bounds is not None:
            theta_lb = lower_bounds[0]
            tau_lb = lower_bounds[1]
            M_lb = lower_bounds[2]
        else:
            theta_lb = 0.01
            tau_lb = 0.01
            M_lb = 0

        theta_bounds = [(theta_lb, None)] * self.n_theta_params
        tau_bounds = [(tau_lb, None)] * self.n_tau_params
        migr_bounds = [(M_lb, None)] * self.n_M_params
        bounds = theta_bounds + tau_bounds + migr_bounds

        return bounds

    def fit(self, S, blocklen=None, initial_values=None, lower_bounds=None, verbose=False):
        """UNTESTED"""
        assert isinstance(S, np.ndarray), "S must be np.ndarray"
        assert S.shape[0] == 3, f"S has wrong shape {S.shape}; there should be 3 sampling states"
        assert S.shape[1] > 0, "S contains no data"

        initial_vals = self.get_initial_values(S, initial_values, blocklen)
        bounds = self.get_bounds(S, lower_bounds)

        return self.minimise_neg_log_likelihood(S, initial_vals, bounds, verbose=verbose)
    
    def fit_resampled_ivs(self):
        raise NotImplementedError("Fitting with sampled IVs not yet implemented.")
    

    @staticmethod
    def composite_likelihood(params, S, epochs, verbose=False):
        """UNTESTED"""
        likelihood_matrix = LikelihoodMatrix(params, S, epochs)
        negll = np.sum(S * likelihood_matrix.matrix)

        if verbose:
            print(params, negll)

        return negll

    @staticmethod
    def pretty_output(inferred_params, negll, aic):
        """TODO: FittedModel class of some sort"""
        df = pd.DataFrame({"Parameter":inferred_params.keys(), "Value":inferred_params.values()})
        df = df.append({"Parameter": "Neg. log-likelihood", "Value": negll}, ignore_index=True)
        df = df.append({"Parameter": "Akaike information crit.", "Value": aic}, ignore_index=True)

        return df
        
    def minimise_neg_log_likelihood(self, S, initial_values, bounds, verbose=False):
        """UNTESTED"""
        opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
        for algo_idx, algo in enumerate(opt_algos):
            optimised = scipy.optimize.minimize(self.composite_likelihood,
                                                x0=initial_values,
                                                method=algo,
                                                args=(S, self.epochs, verbose),
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
        inferred_params = optimised.x
        n_params = len(initial_values) # need to combine these back to make human readable output
        negll = optimised.fun
        aic = 2*n_params + 2*negll

        return inferred_params, negll, n_params, aic
