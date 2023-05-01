import numpy as np
from scipy import linalg
import math

class GeneratorMatrix():
    """
    Generates Q stochastic matrix, which describes transition rates between states between times present and tau1.
    :param float m1_prime: Migration rate from population 1 to population 2
    :param float m2_prime: Migration rate from population 2 to population 1
    :param float c1: Relative size of population 1; see documentation for population sizes
    :param float c2: Relative size of population 2; see documentation for population sizes
    :return: sympy.Matrix
    """
    def __init__(self, matrix_type, theta0=None, theta1=None, theta2=None, theta1_prime=None, theta2_prime=None,
                  m1_star=None, m2_star=None, m1_prime_star=None, m2_prime_star=None):
        
        self.matrix_type = str(matrix_type).upper()
        assert self.matrix_type in ["Q1", "Q2", "Q3"]
        assert theta1 is not None and theta1 > 0, f"invalid value {theta1} for theta1"

        if matrix_type == "Q1":
            assert theta1_prime is not None and theta1_prime > 0, f"invalid value {theta1_prime} for theta1_prime"
            assert theta2_prime is not None and theta2_prime > 0, f"invalid value {theta2_prime} for theta2_prime"
            assert m1_prime_star is not None
            assert m2_prime_star is not None
            self.mig_rate1 = float(m1_prime_star/theta1_prime)
            self.mig_rate2 = float(m2_prime_star/theta2_prime)
            self.pop_size1 = float(theta1_prime/theta1)
            self.pop_size2 = float(theta2_prime/theta1)
        elif matrix_type == "Q2":
            assert theta2 is not None and theta2 > 0, f"invalid value {theta2} for theta2"
            assert m1_star is not None
            assert m2_star is not None
            self.mig_rate1 = float(m1_star)
            self.mig_rate2 = float(m2_star/(theta2/theta1))
            self.pop_size1 = float(1)
            self.pop_size2 = float(theta2/theta1)
        else:
            assert matrix_type == "Q3"
            assert theta0 is not None and theta0 > 0, f"invalid value {theta0} for theta0"
            self.mig_rate1 = float(0)
            self.mig_rate2 = float(0)
            self.pop_size1 = float(theta0/theta1)
            self.pop_size2 = float(theta0/theta1)

        if any([np.isnan(param) for param in [self.mig_rate1, self.mig_rate2, self.pop_size1, self.pop_size2]]):
            self.matrix = np.zeros(shape=(4,4))
        elif any(param == 0 for param in [self.pop_size1, self.pop_size2]):
            self.matrix = np.zeros(shape=(4,4))
        else:
            assert not np.isnan(self.mig_rate1)
            assert not np.isnan(self.mig_rate2)
            assert not np.isnan(self.pop_size1)
            assert not np.isnan(self.pop_size2)
            self.matrix = self.generate()

        assert not np.isnan(np.sum(self.matrix)), f"{self.pop_size1}, {self.pop_size2}, {self.mig_rate1}, {self.mig_rate2}"

    def __repr__(self):
        return str(self.matrix)


    def __getitem__(self, index):
        return self.matrix[index]
    
    def sum(self):
        return np.sum(self.matrix)

    def generate(self):
        matrix = np.array([[-(1/self.pop_size1 + self.mig_rate1), 0, self.mig_rate1, 1/self.pop_size1],
                          [0, -(1/self.pop_size2 + self.mig_rate2), self.mig_rate2, 1/self.pop_size2],
                          [self.mig_rate2/2, self.mig_rate1/2, -((self.mig_rate1+self.mig_rate2)/2), 0],
                          [0,0,0,0]])
        return matrix


    def eigen(self):


        #if round(self.mig_rate1, 6) == 0 and round(self.mig_rate2, 6) == 0 and self.matrix_type in ['Q1', 'Q2']:
        #     eigenvals = np.array([-1/self.pop_size1, -1/self.pop_size2, 0, 0])
        #     eigenvects = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
        #else:
        eigenvals, eigenvects = linalg.eig(self.matrix, left=True, right=False)
        eigenvals = np.sort(eigenvals)
        eigenvects = np.transpose(eigenvects)
        return np.real(eigenvects), np.real(eigenvals)