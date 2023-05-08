import numpy as np
from scipy import linalg
import math

# TODO: rename to TransitionRateMatrix, introduce class StochasticMatrix
class GeneratorMatrix:

    def __init__(self, matrix_type, theta0=None, theta1=None, theta2=None, theta1_prime=None, theta2_prime=None,
                  m1_star=None, m2_star=None, m1_prime_star=None, m2_prime_star=None):
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
        assert theta1 is not None and theta1 > 0, f"invalid value {theta1} for theta1"

        if matrix_type == "Q1":
            assert theta1_prime is not None and theta1_prime > 0, f"invalid value {theta1_prime} for theta1_prime"
            assert theta2_prime is not None and theta2_prime > 0, f"invalid value {theta2_prime} for theta2_prime"
            assert m1_prime_star is not None
            assert m2_prime_star is not None
            c1 = theta1_prime/theta1
            c2 = theta2_prime/theta1
            m1_prime = m1_prime_star/c1
            m2_prime = m2_prime_star/c2
            self.matrix = self.generate_q1(c1, c2, m1_prime, m2_prime)
        elif matrix_type == "Q2":
            assert theta2 is not None and theta2 > 0, f"invalid value {theta2} for theta2"
            assert m1_star is not None
            assert m2_star is not None
            b = theta2/theta1
            m1 = m1_star
            m2 = m2_star/b
            self.matrix = self.generate_q2(b, m1, m2)
        else:
            assert matrix_type == "Q3"
            assert theta0 is not None and theta0 > 0, f"invalid value {theta0} for theta0"
            a = theta0/theta1
            self.matrix = self.generate_q3(a)

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
    
    @staticmethod
    def generate_q1(c1, c2, m1_prime, m2_prime):
        return np.array([[-(1/c1 + m1_prime), 0, m1_prime, 1/c1],
                          [0, -(1/c2 + m2_prime), m2_prime, 1/c2],
                          [m2_prime/2, m1_prime/2, -((m1_prime+m2_prime)/2), 0],
                          [0,0,0,0]])
    
    @staticmethod
    def generate_q2(b, m1, m2):
        return np.array([[-(1 + m1), 0, m1, 1],
                          [0, -(1/b + m2), m2, 1/b],
                          [m2/2, m1/2, -((m1+m2)/2), 0],
                          [0,0,0,0]])
    
    @staticmethod
    def generate_q3(a):
        return np.array([[-1/a, 0, 0, 1/a],
                         [0, -1/a, 0, 1/a],
                         [0, 0, -1/a, 1/a],
                         [0,0,0,0]])


    def eigen(self):
        """Calculate the eigenvalues and left eigenvectors of the generator matrix."""
        t = self.transpose()
        eigenvals, eigenvects = linalg.eig(t) # left eigenvects = right eigenvects of transposed matrix
        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvects = eigenvects[:,sorted_indices]
        return np.real(eigenvects.T), np.real(eigenvals)
    
    @staticmethod
    def from_params(params):
        """Generate Q1, Q2, and Q3 from a list of parameters [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]"""
        theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star = params
        q1 = GeneratorMatrix(matrix_type="Q1", theta1=theta1, theta1_prime=theta1_prime, theta2_prime=theta2_prime,
                              m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
        q2 = GeneratorMatrix(matrix_type="Q2", theta1=theta1, theta2=theta2, m1_star=m1_star, m2_star=m2_star)
        q3 = GeneratorMatrix(matrix_type="Q3", theta1=theta1, theta0=theta0)

        return q1, q2, q3
