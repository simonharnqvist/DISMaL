import numpy as np
from stochastic_matrix import StochasticMatrix
from scipy import linalg
from scipy.stats import poisson
# TODO: reimplement with Cython; likely to make the greatest difference
class LikelihoodMatrix:
    
    def __init__(self, q1, q2, q3, t1, v, S=None, s_vals=None):
        """S is a matrix of counts; return matrix of likelihood of params given s (columns) and state (rows)"""
        
        if s_vals is None:
            assert S is not None
            s_vals = [s for s in range(0, S.shape[1])]
        self.s_vals = s_vals

        self.t0 = t1+v
        self.t1 = t1

        self.g = q1.eigenvectors
        self.c = q2.eigenvectors
        self.ginv = q1.eigenvectors_inv
        self.cinv = q2.eigenvectors_inv
        self.alpha = -q1.eigenvalues[0:3]
        self.beta = -q2.eigenvalues[0:3]
        self.gamma = -q3[0,0]

        self.gg = -self.ginv @ np.diag(self.g[:,3])
        self.cc = -self.cinv @ np.diag(self.c[:,3])
        self.pij1 = StochasticMatrix(eigenvect_mat=self.g, inv_eigenvect_mat=self.ginv, eigenvals=q1.eigenvalues, t=t1).matrix
        self.pij2 = StochasticMatrix(eigenvect_mat=self.c, inv_eigenvect_mat=self.cinv, eigenvals=q2.eigenvalues, t=v).matrix

        self.matrix = self.generate()

    def __repr__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def alpha_matrix(self, rel_mu=1):
        """Generate matrix of s values (columns) x adjusted alpha values (rows). This corresponds to P(s events before coalescence) * 1-P(s events)"""
        alpha, s_vals = np.array(self.alpha), np.array(self.s_vals)
        alphas = []
        for s in s_vals:
            alphas.append((alpha/(alpha+rel_mu)) * ((rel_mu/(alpha+rel_mu))**s) * (1-poisson.cdf(s, (self.t1*(alpha+rel_mu)))))
        return np.transpose(np.array(alphas))

    def beta_matrix(self, rel_mu=1):
        beta, s_vals = np.array(self.beta), np.array(self.s_vals)
        betas = []
        for s in s_vals:
            betas.append((beta/(beta+rel_mu)) * ((rel_mu/(beta+rel_mu))**s) *
                        np.exp(beta*self.t1) * (poisson.cdf(s, (self.t1*(beta+rel_mu))) - poisson.cdf(s, (self.t0*(beta+rel_mu)))))
        return np.transpose(np.array(betas))

    def gamma_matrix(self, rel_mu=1):
        gamma, s_vals = np.array(self.gamma), np.array(self.s_vals)
        gammas = []
        for s in s_vals:
            gammas.append((gamma/(gamma+rel_mu)) * ((rel_mu/(gamma+rel_mu))**s) *
                        np.exp(gamma*self.t0) * poisson.cdf(s, (self.t0*(gamma+rel_mu))))
        return np.transpose(np.array(gammas))

    
    def generate(self):

        ll_matrix = [-np.log(self.gg[i, 0:3] @ self.alpha_matrix() + self.pij1[i, 0:3] @ self.cc[0:3, 0:3] @ self.beta_matrix() +
                         (1 - (self.pij1@self.pij2)[i,3]) * self.gamma_matrix()) for i in [0, 1, 2]]
    
        return ll_matrix