import numpy as np
from scipy import linalg
import math

class TransitionRateMatrix:

    def __init__(self, thetas, Ms, symmetric_migration=False):
        """Create a generator matrix that describes that transition rates between states in the stochastic model."""

        self.pop_size1 = thetas[0]
        self.pop_size2 = thetas[1]
        self.mig_rate1 = Ms[0]/thetas[0]

        if symmetric_migration is True:
            self.mig_rate2 = self.mig_rate1
        elif len(Ms) == 1:
            self.mig_rate2 = 0
        else:
            self.mig_rate2 = Ms[1]/thetas[1]

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
        return np.array([
            [-(1/self.pop_size1 + self.mig_rate1), 0, self.mig_rate1, 1/self.pop_size1],
            [0, -(1/self.pop_size2 + self.mig_rate2), self.mig_rate2, 1/self.pop_size2],
            [self.mig_rate2/2, self.mig_rate1/2, - (self.mig_rate1+self.mig_rate2)/2, 0],
            [0, 0, 0, 0]
        ])

    def eigen(self):
        """Calculate the eigenvalues and left eigenvectors of the generator matrix."""
        t = self.transpose()
        # left eigenvects = right eigenvects of transposed matrix
        eigenvals, eigenvects = linalg.eig(t)
        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvects = eigenvects[:, sorted_indices]
        return np.real(eigenvects.T), np.real(eigenvals)

class StochasticMatrix:
    
    def __init__(self, eigenvect_mat, inv_eigenvect_mat, eigenvals, t):
        """ Fast matrix exponentiation to produce stochastic matrix from eigenvalues and eigenvectors of transition rate matrix. Equivalent to linalg.expm(Q), but faster if eigenvalues+vectors are already available.

        Args:
            matrix (_type_): _description_
            inv_matrix (_type_): _description_
            eigenvalues (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.eigenvect_mat = eigenvect_mat
        self.inv_eigenvect_mat = inv_eigenvect_mat
        self.eigenvals = eigenvals
        self.t = t
        self.matrix = self.generate()

    def __repr__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def generate(self):
        return self.inv_eigenvect_mat @ np.diag(np.exp(self.eigenvals * self.t)) @ self.eigenvect_mat