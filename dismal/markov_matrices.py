import numpy as np
from scipy import linalg
import math
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class TransitionRateMatrix:

    def __init__(self, single_deme, thetas, ms, asymmetric_migration=True):
        """Generator matrix that describes that transition rates between states in the stochastic model.

        Args:
            single_deme (bool): Whether to construct matrix for single deme (i.e. in final epoch).
            thetas (iterable): Theta (population size) parameters.
            ms (iterable): Migration rates.
            asymmetric_migration (bool, optional): _description_. Defaults to True.
        """

        assert 1 <= len(thetas) <= 2
        assert 0 <= len(ms) <= 2

        self.single_deme = single_deme # q3?

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
        """Generate transition rate matrix."""

        if any([self.pop_size1, self.pop_size2, self.mig_rate1, self.mig_rate2]) < 0:
            return None
        elif any(math.isnan(param) for param in [self.pop_size1, self.pop_size2, self.mig_rate1, self.mig_rate2]):
            return None
        elif any(math.isinf(param) for param in [self.pop_size1, self.pop_size2, self.mig_rate1, self.mig_rate2]):
            return None
        elif self.single_deme:
            return np.array([
                [-(1/self.pop_size1), 0, 0, 1/self.pop_size1],
                [0, -(1/self.pop_size1), 0, 1/self.pop_size1],
                [0, 0, - (1/self.pop_size1), 1/self.pop_size1],
                [0, 0, 0, 0]
            ])
        else:
            return np.array([
            [-(1/self.pop_size1 + self.mig_rate1), 0, self.mig_rate1, 1/self.pop_size1],
            [0, -(1/self.pop_size2 + self.mig_rate2), self.mig_rate2, 1/self.pop_size2],
            [self.mig_rate2/2, self.mig_rate1/2, - (self.mig_rate1+self.mig_rate2)/2, 0],
            [0, 0, 0, 0]
        ])

    def eigen(self):
        """Calculate the eigenvalues and left eigenvectors of the generator matrix."""
        eigenvals, eigenvects = linalg.eig(self.transpose())
        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvects = eigenvects[:, sorted_indices]
        v, d = np.real(eigenvects.T), np.real(eigenvals)
        
        return v,d
        

class StochasticMatrix:
    
    def __init__(self, Q, t):
        """Transition probability (stochastic) matrix of demography during single epoch.

        Args:
            Q (TransitionRateMatrix): Transition rate matrix.
            t (float): Epoch duration in Ne generations.
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
        """Generate stochastic matrix."""
        return self.inv_eigenvect_mat @ np.diag(np.exp(self.eigenvals * self.t)) @ self.eigenvect_mat