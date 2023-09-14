import numpy as np
from scipy import linalg
import math

class TransitionRateMatrix:

    def __init__(self, thetas, ms, asymmetric_migration=True):
        """Create a generator matrix that describes that transition rates between states in the stochastic model."""

        assert 1 <= len(thetas) <= 2
        assert 0 <= len(ms) <= 2

        self.pop_size1 = thetas[0]

        if len(thetas) == 2:
            self.pop_size2 = thetas[1]
        else:
            self.pop_size2 = self.pop_size1
        
        if len(ms) > 0:
            self.mig_rate1 = ms[0]
        else:
            self.mig_rate1 = 0

        if asymmetric_migration is False:
            self.mig_rate2 = self.mig_rate1
        elif len(ms) == 1:
            self.mig_rate2 = 0
        else:
            self.mig_rate2 = ms[1]

        self.matrix = self.generate()

        self.eigenvectors, self.eigenvalues = self.eigen()
        self.eigenvectors_inv = linalg.inv(self.eigenvectors)


    def __repr__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def sum(self):
        return np.sum(self.matrix)

    def transpose(self):
        return np.transpose(self.matrix)

    def generate(self):

        if any([self.pop_size1, self.pop_size2, self.mig_rate1, self.mig_rate2]) < 0:
            return None
        elif any(math.isnan(param) for param in [self.pop_size1, self.pop_size2, self.mig_rate1, self.mig_rate2]):
            return None
        elif any(math.isinf(param) for param in [self.pop_size1, self.pop_size2, self.mig_rate1, self.mig_rate2]):
            return None
        else:
            return np.array([
            [-(1/self.pop_size1 + self.mig_rate1), 0, self.mig_rate1, 1/self.pop_size1],
            [0, -(1/self.pop_size2 + self.mig_rate2), self.mig_rate2, 1/self.pop_size2],
            [self.mig_rate2/2, self.mig_rate1/2, - (self.mig_rate1+self.mig_rate2)/2, 0],
            [0, 0, 0, 0]
        ])

    def eigen(self):
        """Calculate the eigenvalues and left eigenvectors of the generator matrix."""
        t = self.transpose() # left eigenvects = right eigenvects of transposed matrix
        eigenvals, eigenvects = linalg.eig(t)
        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvects = eigenvects[:, sorted_indices]
        return np.real(eigenvects.T), np.real(eigenvals)

class StochasticMatrix:
    
    def __init__(self, Q, t):
        """ Fast matrix exponentiation to produce stochastic matrix from eigenvalues and eigenvectors of transition rate matrix. Equivalent to linalg.expm(Q), but faster if eigenvalues+vectors are already available.
    """
        self.eigenvect_mat = Q.eigenvectors
        self.inv_eigenvect_mat = Q.eigenvectors_inv
        self.eigenvals = Q.eigenvalues
        self.t = t
        self.matrix = self.generate()

    def __repr__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def generate(self):
        return self.inv_eigenvect_mat @ np.diag(np.exp(self.eigenvals * self.t)) @ self.eigenvect_mat