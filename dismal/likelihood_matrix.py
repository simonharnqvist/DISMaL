import numpy as np
from dismal.markov_matrices import TransitionRateMatrix, StochasticMatrix
from dismal.demography import Epoch
from scipy import linalg
from scipy.stats import poisson


class LikelihoodMatrix:

    def __init__(self, params, S, epoch_objects):
        """S is a matrix of counts; return matrix of likelihood of 
        params given s (columns) and state (rows)"""

        self.S = np.array(S)
        self.s_vals = [s for s in range(0, self.S.shape[1])]

        assert isinstance(epoch_objects[0], Epoch)

        self.epochs_post_div = len(epoch_objects)-1
        self.n_populations = self.epochs_post_div*2 + 1
        self.thetas = params[0:self.n_populations]

        self.taus = [params[self.n_populations]]
        if self.epochs_post_div == 2:
            self.v = params[self.n_populations+1]
            self.tau1 = self.taus[0] - self.v
            self.taus.append(self.tau1)

        self.taus = params[self.n_populations:(self.n_populations+self.epochs_post_div)]
        self.Ms = params[(self.n_populations+self.epochs_post_div):]
        
        self.thetas_iter = iter(self.thetas[1:]) # exclude first epoch since no gene flow
        self.Ms_iter = iter(self.Ms)

        self.Qs = []
        for epoch_obj in epoch_objects[1:]: 
            self.Qs.append(self.generate_trm(epoch_obj))

        self.Ps = []
        for idx, Q in enumerate(self.Qs):
            self.Ps.append(StochasticMatrix(Q.eigenvectors, Q.eigenvectors_inv, Q.eigenvalues, self.taus[idx]))

        if self.epochs_post_div == 1:
            self.matrix = self.two_stage_ll_matrix()
        elif self.epochs_post_div == 2:
            self.matrix = self.three_stage_ll_matrix()
        else:
            raise ValueError(f"Wrong number of post-divergence epochs {self.epochs_post_div}")
        
    def __repr__(self):
        return str(self.matrix)

    def __str__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def generate_trm(self, epoch_obj):
        epoch_thetas = [next(self.thetas_iter), next(self.thetas_iter)]
        epoch_Ms = [next(self.Ms_iter) for _ in range(epoch_obj.n_M_params)]

        Q = TransitionRateMatrix(thetas = epoch_thetas,
                                Ms = epoch_Ms,
                                symmetric_migration=epoch_obj.symmetric_migration)
        
        return Q
    
    @classmethod
    def poisson_cdf(self, s, time, rate_param):

        if time is None:
            return 0
        else:
            return poisson.cdf(s, (time*(rate_param)))
    
    @classmethod
    def diagonal_matrix_mult(self, mat, mat_inv):
        return -mat_inv @ np.diag(mat[:, 3])

    def eigenvalue_matrix(self, eigenvals, start_time, end_time, rel_mu=1):
        """Generate matrix of s values (columns) x adjusted eigenvalues (rows).
        This corresponds to P(s events before coalescence) * 1-P(s events)"""
        
        if len(eigenvals) > 1:   
            lmbda = -np.array(eigenvals)[0:3]
        else:
            lmbda = -np.array(eigenvals)[0]
        
        s_vals = np.array(self.s_vals)

        eigv_mat = [(lmbda/(lmbda+rel_mu))
                    * ((rel_mu/(lmbda+rel_mu))**s)
                    * np.exp(lmbda*start_time)
                    * (self.poisson_cdf(s, time=start_time, rate_param=(lmbda+rel_mu))
                    - self.poisson_cdf(s, time=end_time, rate_param=(lmbda+rel_mu)))
                    for s in s_vals]
        return np.transpose(np.array(eigv_mat))

    def two_stage_ll_matrix(self):
        return -np.log(np.array([np.identity(4)[i, 0:3]
                                 @ self.diagonal_matrix_mult(self.Qs[0].eigenvectors, self.Qs[0].eigenvectors_inv)[0:3, 0:3]
                                 @ self.eigenvalue_matrix(eigenvals=self.Qs[0].eigenvalues,
                                                          start_time=0, end_time=self.taus[0])
                                 + (1 - (np.identity(4)@self.Ps[0].matrix)[i, 3])
                                 * self.eigenvalue_matrix(eigenvals=[-1/self.thetas[0]],
                                                           start_time=self.taus[0], end_time=None)
                                                             for i in [0, 1, 2]]))

    def three_stage_ll_matrix(self):
        return -np.log([np.array(
            self.eigenvalue_matrix(eigenvals=self.Qs[0].eigenvalues, start_time=0, end_time=self.taus[1])
            + (self.Ps[0][i,0:3] 
               @ self.diagonal_matrix_mult(self.Qs[1].eigenvectors, self.Qs[1].eigenvectors_inv)[0:3, 0:3] 
               @ self.eigenvalue_matrix(eigenvals=self.Qs[1].eigenvalues, start_time=self.taus[1], end_time=self.taus[0]))
            + (1-(self.Ps[0].matrix@self.Ps[1].matrix)[i,3]) )
            * self.eigenvalue_matrix([-1/self.thetas[0]], start_time=self.taus[0], end_time=None)
                        for i in [0, 1, 2]])


    def log_likelihood_array(self, state_idx):

        if state_idx == 0 or state_idx == 1:
            self.eigenvalue_matrix(eigenvals=self.Qs[state_idx].eigenvalues, start_time=self.taus[])