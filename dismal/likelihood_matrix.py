import numpy as np
from stochastic_matrix import StochasticMatrix
from generator_matrices import GeneratorMatrix
from scipy import linalg
from scipy.stats import poisson
# TODO: reimplement with Cython; likely to make the greatest difference
class LikelihoodMatrix:
    
    def __init__(self, params, S):
        """S is a matrix of counts; return matrix of likelihood of params given s (columns) and state (rows)"""


        s_vals = [s for s in range(0, S.shape[1])]
        self.s_vals = s_vals

        for mig_param in ["M1", "M2", "M1_star", "M2_star", "M1_prime_star", "M2_prime_star"]:
            if mig_param not in list(params.keys()):
                params[mig_param] = 0

        if "t1" not in list(params.keys()): # two stage model

            q_anc = GeneratorMatrix(matrix_type="Q3", theta1=params["theta1"], theta0=params["theta0"])
            q_mig = GeneratorMatrix(matrix_type="Q2", theta1=params["theta1"], theta2=params["theta2"],
                                     m1_star=params["M1"], m2_star=params["M2"])
            self.c = q_mig.eigenvectors
            self.cinv = q_mig.eigenvectors_inv
            self.beta = -q_mig.eigenvalues[0:3]
            self.gamma = -q_anc[0,0]

            self.t0 = params["t0"]
            self.cc = -self.cinv @ np.diag(self.c[:,3])
            self.pij = linalg.expm(q_mig.matrix*self.t0)

            self.matrix = self.two_stage_ll_matrix()

        else: # three stage model
            q1 = GeneratorMatrix(matrix_type="Q1", theta1=params["theta1"], theta1_prime=params["theta1_prime"],
                                  theta2_prime=params["theta2_prime"], m1_prime_star=params["M1_prime_star"],
                                    m2_prime_star=params["M2_prime_star"])
            q2 = GeneratorMatrix(matrix_type="Q2", theta1=params["theta1"], theta2=params["theta2"],
                                 m1_star=params["M1_star"], m2_star=params["M2_star"])
            q3 = GeneratorMatrix(matrix_type="Q3", theta1=params["theta1"], theta0=params["theta0"])

            self.g = q1.eigenvectors
            self.c = q2.eigenvectors
            self.ginv = q1.eigenvectors_inv
            self.cinv = q2.eigenvectors_inv
            self.alpha = -q1.eigenvalues[0:3]
            self.beta = -q2.eigenvalues[0:3]
            self.gamma = -q3[0,0]

            self.gg = -self.ginv @ np.diag(self.g[:,3])
            self.cc = -self.cinv @ np.diag(self.c[:,3])

            self.t1 = params["t1"]
            self.v = params["v"]
            self.t0 = params["t1"] + params["v"]
            self.pij1 = linalg.expm(q1.matrix * self.t1)
            self.pij2 = linalg.expm(q1.matrix * self.t1) * linalg.expm(q2.matrix * self.v)

            self.matrix = self.three_stage_ll_matrix()

    def __repr__(self):
        return str(self.matrix)
    
    def __str__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]
    
    def alpha_matrix(self, t1, rel_mu=1):
        """Generate matrix of s values (columns) x adjusted alpha values (rows). This corresponds to P(s events before coalescence) * 1-P(s events)"""
        alpha, s_vals = np.array(self.alpha), np.array(self.s_vals)
        alphas = []
        for s in s_vals:
            alphas.append((alpha/(alpha+rel_mu)) * ((rel_mu/(alpha+rel_mu))**s) * (1-poisson.cdf(s, (t1*(alpha+rel_mu)))))
        return np.transpose(np.array(alphas))

    def beta_matrix(self, t1, t0, rel_mu=1):
        beta, s_vals = np.array(self.beta), np.array(self.s_vals)
        betas = []
        for s in s_vals:
            betas.append((beta/(beta+rel_mu)) * ((rel_mu/(beta+rel_mu))**s) *
                        np.exp(beta*t1) * (poisson.cdf(s, (t1*(beta+rel_mu))) - poisson.cdf(s, (t0*(beta+rel_mu)))))
        return np.transpose(np.array(betas))

    def gamma_matrix(self, t0, rel_mu=1):
        gamma, s_vals = np.array(self.gamma), np.array(self.s_vals)
        gammas = []
        for s in s_vals:
            gammas.append((gamma/(gamma+rel_mu)) * ((rel_mu/(gamma+rel_mu))**s) *
                        np.exp(gamma*t0) * poisson.cdf(s, (t0*(gamma+rel_mu))))
        return np.transpose(np.array(gammas))

    def two_stage_ll_matrix(self):
        return np.array([-np.log(np.identity(4)[i, 0:3] @ self.cc[0:3, 0:3] @ self.beta_matrix(t1=0, t0=self.t0) +
                         (1 - (np.identity(4)@self.pij)[i,3]) * self.gamma_matrix(t0=self.t0)) for i in [0, 1, 2]])
    
    def three_stage_ll_matrix(self):
        return np.array([-np.log(self.gg[i, 0:3] @ self.alpha_matrix(t1=self.t1) + self.pij1[i, 0:3] @
                                  self.cc[0:3, 0:3] @ self.beta_matrix(t1=self.t1, t0=self.t0) +
                         (1 - (self.pij1@self.pij2)[i,3]) * self.gamma_matrix(t0=self.t0)) for i in [0, 1, 2]])