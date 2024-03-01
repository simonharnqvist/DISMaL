import numpy as np
from collections import Counter
from scipy.stats import poisson
from functools import reduce
import warnings
from dismal.markov_matrices import StochasticMatrix, TransitionRateMatrix
from dismal.demography import Epoch
from dismal.print_results import print_output
from dismal.modelsimulation import ModelSimulation

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)


class ModelInstance:
    """Class to represent single model object with given parameter values"""

    def __init__(self, param_vals, epochs):

        self.epoch_templates = epochs

        self.param_vals = param_vals
        self.n_theta_params = np.sum([epoch.n_thetas for epoch in self.epoch_templates])
        self.n_epoch_durations = len(list(self.epoch_templates)) - 1
        self.n_mig_params = np.sum([epoch.n_mig_params for epoch in self.epoch_templates])

        self._check_parameter_values()

        self.thetas = param_vals[0:self.n_theta_params]
        self.epoch_durations = param_vals[self.n_theta_params:
                                          (self.n_theta_params + self.n_epoch_durations)]
        self.migration_rates = param_vals[(self.n_theta_params + self.n_epoch_durations):]

        self.start_times = [sum(self.epoch_durations[0:i]) 
                            for i in range(len(self.epoch_durations)+1)]
        self.end_times = [sum(self.epoch_durations[0:i]) 
                     for i in range(1, len(self.epoch_durations)+1)]
        self.end_times.extend([None])

        self.epochs = self._add_params_to_epochs()
        self.deme_ids = [epoch.deme_ids for epoch in self.epochs]

        self.Qs = self._generate_markov_chain()
        self.Ps = self._generate_stochastic_matrices()
        self.QQs = self._generate_QQs()
        self.eigenvals = self._calculate_eigenvals()

        self.obs_s1 = None
        self.obs_s2 = None
        self.obs_s3 = None
        self.expt_s1 = None
        self.expt_s2 = None
        self.expt_s3 = None

        self.negll = None
        self.claic = None


    def _check_parameter_values(self):
        expected_n_params = self.n_theta_params + self.n_epoch_durations + self.n_mig_params
        assert len(self.param_vals) == expected_n_params, f"""Wrong number of parameters -- expected {expected_n_params} parameters:
            {self.n_theta_params} thetas
            {self.n_epoch_durations} epoch durations
            {self.n_mig_params} migration rates
        
        Found {len(self.param_vals)} parameters
        
        Please check your parameter values or respecify the model"""


    def _add_params_to_epochs(self):
        """Assign parameter value to epochs"""
        thetas_iter = iter(self.thetas)
        if self.migration_rates is not None:
            ms_iter = iter(self.migration_rates)

        updated_epochs = []

        for epoch_idx, epoch in enumerate(self.epoch_templates):
            updated_epoch = Epoch(
                n_demes=epoch.n_demes,
                deme_ids=epoch.deme_ids,
                migration=epoch.migration,
                asymmetric_migration=epoch.asymmetric_migration,
                migration_direction=epoch.migration_direction)

            updated_epoch.thetas = [next(thetas_iter) for _ in range(epoch.n_thetas)]
            updated_epoch.start_time = self.start_times[epoch_idx]
            updated_epoch.end_time = self.end_times[epoch_idx]
            if updated_epoch.migration is True:
                updated_epoch.migration_rates = [next(ms_iter) 
                                                 for _ in range(epoch.n_mig_params)]

            updated_epochs.append(updated_epoch)

        return updated_epochs


    def _generate_markov_chain(self):
        """Generate transition rate matrices for a given model given parameter values"""
        Qs = []

        thetas_iter = iter(self.thetas)
        ms_iter = iter(self.migration_rates)

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

        Qs.append(TransitionRateMatrix(single_deme=True, 
                                       thetas=[self.thetas[-1]], ms=[0]))

        return Qs


    def _generate_stochastic_matrices(self):
        """Generate P matrices from Q matrices"""
        return [StochasticMatrix(Q, t=self.epoch_durations[idx]) 
                   for idx, Q in enumerate(self.Qs[:-1])]


    def _generate_QQs(self):
        """Generate QQ matrices"""
        return [-Q.eigenvectors_inv 
                    @ np.diag(Q.eigenvectors[:, -1]) 
                    for Q in self.Qs[:-1]]


    def _calculate_eigenvals(self):
        eigenvals = [np.array(-Q.eigenvalues[0:3]) for Q in self.Qs[:-1]]
        eigenvals.append(np.array([self.Qs[-1][0, 3]]))
        return eigenvals
 

    @staticmethod
    def _poisson_cdf(s_max, t, eigenvalues):
        """Cumulative Poisson probability distribution of segregating sites.

        Args:
            s_max (int): Maximum s-value to calculate
            t (float): Epoch duration
            eigenvalues (ndarray): Eigenvalues of transition rate matrix of epoch.

        Returns:
            ndarray: Matrix n(eigenvalues) x n(s) of PoisCDF(s).
        """
        if t is None:
            return np.zeros(shape=(len(eigenvalues), s_max+1))
        elif t == 0:
            return np.ones(shape=(len(eigenvalues), s_max+1))
        else:
            return np.transpose([poisson.cdf(s, t*(eigenvalues+1)) for s in range(s_max+1)])
                

    @staticmethod
    def _transform_eigenvalues_s(s_max, eigenvalues, start_time, end_time):
        """Transform s counts by eigenvalues to generate n(eigenvalues) x n(s) matrix"

        Args:
            s_max (int): Maximum s-value
            eigenvalues (ndarray): Eigenvalues of transition rate matrix of epoch.
            start_time (float): Epoch start time in Ne generations.
            end_time (float): Epoch end time in Ne generations.

        Returns:
            ndarray: Matrix n(eigenvalues) x n(s) of s-transformed eigenvalues.
        """

        transf = np.transpose([(eigenvalues/(eigenvalues+1)) 
                            * (1/(eigenvalues+1)) ** s_val for s_val in range(s_max + 1)])
        
        eigenvalues_exp = np.exp(eigenvalues * start_time)
        pois_start = ModelInstance._poisson_cdf(s_max, start_time, eigenvalues)
        pois_end = ModelInstance._poisson_cdf(s_max, end_time, eigenvalues)

        return np.transpose(np.transpose(transf) * eigenvalues_exp * np.transpose((pois_start - pois_end)))


    def pr_s(self, s_max, state, neglog = False):   
        """Calculate Pr(S = s) for all values up to s_max, in a given sampling state {1,2,3}"""

        state_idx = state - 1
        assert state in [1,2,3]

        As = [self._transform_eigenvalues_s(s_max, 
                                            self.eigenvals[epoch_idx], 
                                            self.start_times[epoch_idx], 
                                            self.end_times[epoch_idx]) 
                                            for epoch_idx in range(len(self.epochs))]

        first_epoch = self.QQs[0][state_idx, 0:-1] @ As[0]

        if len(self.epochs) > 2:
            middle_epochs = [self.Ps[epoch_idx-1][state_idx, 0:-1] 
                             @ self.QQs[epoch_idx][0:-1, 0:-1] 
                             @ As[epoch_idx] for epoch_idx in range(1, len(self.epochs)-1)]
        else:
            middle_epochs = np.zeros(shape=(1, s_max+1))

        last_epoch = ((1-(reduce(np.matmul, [p[:] for p in self.Ps]))[state_idx,-1]) * As[-1])

        ps_array = reduce(np.add, [first_epoch, middle_epochs, last_epoch])[0]

        if neglog is True:
            ps_array = -np.log(ps_array)
        
        return ps_array
    

    def expected_s1(self, s_max, neglog=False):
        """Convenience function for S1 distribution"""
        return self.pr_s(s_max, state=1, neglog=neglog)
    

    def expected_s2(self, s_max, neglog=False):
        """Convenience function for S2 distribution"""
        return self.pr_s(s_max, state=2, neglog=neglog)
    

    def expected_s3(self, s_max, neglog=False):
        """Convenience function for S3 distribution"""
        return self.pr_s(s_max, state=3, neglog=neglog)
    
    
    def neg_composite_log_likelihood(self, s1, s2, s3):
        """Calculate negative composite log-likelihood for dataset"""

        self.obs_s1 = np.array(s1)
        self.obs_s2 = np.array(s2)
        self.obs_s3 = np.array(s3)

        s_maxs = [s.shape[0]-1 for s in [self.obs_s1, self.obs_s2, self.obs_s3]]
        self.expt_s1, self.expt_s2, self.expt_s3 = [self.pr_s(s_maxs[state-1], 
                                                              state, neglog=True) for state in [1,2,3]]
        
        lnl = np.sum([
            np.sum(self.obs_s1 * self.expt_s1),
            np.sum(self.obs_s2 * self.expt_s2),
            np.sum(self.obs_s3 * self.expt_s3)])
        
        self.negll = lnl
    
        return lnl
    
    def simulate(self, mutation_rate, blocklen, recombination_rate, blocks_per_state):
        return ModelSimulation(self, mutation_rate=mutation_rate, 
                        blocklen=blocklen, 
                        recombination_rate=recombination_rate,
                        blocks_per_state=blocks_per_state)




