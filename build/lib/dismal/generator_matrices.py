import numpy as np
from scipy import linalg
import math

# TODO: rename to TransitionRateMatrix, introduce class StochasticMatrix


class GeneratorMatrix:

    def __init__(self, matrix_type, theta0=None, theta1=None, theta2=None,
                 theta1_prime=None, theta2_prime=None, m1_star=None,
                 m2_star=None, m1_prime_star=None, m2_prime_star=None):
        """Create a generator matrix that describes that transition rates between states in the stochastic model.

        Args:
            matrix_type (str): One of "Q1", "Q2", "Q3".
            theta0 (float, optional): Parameter value. Defaults to None.
            theta1 (float, optional): Parameter value. Defaults to None.
            theta2 (float, optional): Parameter value. Defaults to None.
            theta1_prime (float, optional): Parameter value. Defaults to None.
            theta2_prime (float, optional): Parameter value. Defaults to None.
            m1_star (float, optional): Parameter value. Defaults to None.
            m2_star (float, optional): Parameter value. Defaults to None.
            m1_prime_star (float, optional): Parameter value. Defaults to None.
            m2_prime_star (float, optional): Parameter value. Defaults to None.
        """

        self.matrix_type = str(matrix_type).upper()
        assert self.matrix_type in ["Q1", "Q2", "Q3"]
        # assert theta1 is not None and theta1 > 0, f"invalid value {theta1} for theta1"

        if matrix_type == "Q1":
            # assert theta1_prime is not None and theta1_prime > 0, f"invalid value {theta1_prime} for theta1_prime"
            # assert theta2_prime is not None and theta2_prime > 0, f"invalid value {theta2_prime} for theta2_prime"
            # assert m1_prime_star is not None
            # assert m2_prime_star is not None
            self.pop_size1 = theta1_prime
            self.pop_size2 = theta2_prime
            self.mig_rate1 = m1_prime_star/theta1_prime
            self.mig_rate2 = m2_prime_star/theta2_prime
        else:
            assert matrix_type == "Q2"
            # assert theta2 is not None and theta2 > 0, f"invalid value {theta2} for theta2"
            # assert m1_star is not None
            # assert m2_star is not None
            self.pop_size1 = theta1
            self.pop_size2 = theta2
            self.mig_rate1 = m1_star/theta1
            self.mig_rate2 = m2_star/theta2

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
            [-(1/self.pop_size1 + self.mig_rate1),
             0, self.mig_rate1, 1/self.pop_size1],
            [0, -(1/self.pop_size2 + self.mig_rate2),
             self.mig_rate2, 1/self.pop_size2],
            [self.mig_rate2/2, self.mig_rate1/2, -
                (self.mig_rate1+self.mig_rate2)/2, 0],
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

    @staticmethod
    def from_params(params):
        """Generate Q1, Q2 from a list of parameters [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]"""
        theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star = params
        q1 = GeneratorMatrix(matrix_type="Q1", theta1=theta1, theta1_prime=theta1_prime, theta2_prime=theta2_prime,
                             m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
        q2 = GeneratorMatrix(matrix_type="Q2", theta1=theta1,
                             theta2=theta2, m1_star=m1_star, m2_star=m2_star)

        return q1, q2
