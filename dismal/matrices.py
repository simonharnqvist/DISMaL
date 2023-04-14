import numpy as np
from scipy import linalg

class GeneratorMatrix():
    """
    Generates Q stochastic matrix, which describes transition rates between states between times present and tau1.
    :param float m1_prime: Migration rate from population 1 to population 2
    :param float m2_prime: Migration rate from population 2 to population 1
    :param float c1: Relative size of population 1; see documentation for population sizes
    :param float c2: Relative size of population 2; see documentation for population sizes
    :return: sympy.Matrix
    """
    def __init__(self, matrix_type, m1=None, m2=None, m1_prime=None, m2_prime=None, a=None, b=None, c1=None, c2=None):
        self.matrix_type = matrix_type

        assert self.matrix_type in ["Q1", "Q2", "Q3"]

        if matrix_type == "Q1":
            assert c1 is not None
            assert c2 is not None
            assert m1_prime is not None
            assert m2_prime is not None
            self.mig_rate1 = float(m1_prime)
            self.mig_rate2 = float(m2_prime)
            self.pop_size1 = float(c1)
            self.pop_size2 = float(c2)
        elif matrix_type == "Q2":
            assert b is not None
            assert m1 is not None
            assert m2 is not None
            self.mig_rate1 = float(m1)
            self.mig_rate2 = float(m2)
            self.pop_size1 = float(1)
            self.pop_size2 = float(b)
        else:
            assert matrix_type == "Q3"
            assert a is not None
            self.mig_rate1 = float(0)
            self.mig_rate2 = float(0)
            self.pop_size1 = float(a)
            self.pop_size2 = float(a)

        if any([np.isnan(param) for param in [self.mig_rate1, self.mig_rate2, self.pop_size1, self.pop_size2]]):
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


    def left_eigenvectors(self):


        if round(self.mig_rate1, 6) == 0 and round(self.mig_rate2, 6) == 0 and self.matrix_type in ['Q1', 'Q2']:
            eigenvals = np.array([-1/self.pop_size1, -1/self.pop_size2, 0, 0])
            eigenvects = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])
        else:
            eigenvals, eigenvects = self._calculate_left_eigenvectors(self.matrix)

        return np.real(eigenvects), np.abs(np.real(eigenvals))
    
    @staticmethod
    def _calculate_left_eigenvectors(matrix):
        mat_transpose = np.transpose(matrix)
        eigenvals, eigenvects = linalg.eig(mat_transpose, left=False, right=True)
        return eigenvals, eigenvects


    @staticmethod
    def inverse(matrix):
        """
        Find inverse of a matrix.
        :param np.array matrix: Matrix to be inverted
        :return: sp.Matrix
        """
        return linalg.inv(matrix)




class TransitionMatrix:
    """Probability matrix P; generator_matrices is [Q1, Q2, Q3]"""

    def __init__(self, q1, q2, q3, t, tau1, tau0):
#
        self.q1 = q1.matrix
        self.q2 = q2.matrix
        self.q3 = q3.matrix
        self.t = t
        self.tau1 = tau1
        self.tau0 = tau0
        self.matrix = self.generate()

    def __repr__(self):
        return str(self.matrix)


    def __getitem__(self, index):
        return self.matrix[index]
    
    def sum(self):
        return np.sum(self.matrix)

    def generate(self):
        #assert np.isnan(np.sum(self.q1)), "NaN values in Q1 matrix"
        #assert np.isnan(np.sum(self.q2)), "NaN values in Q2 matrix"
        #assert np.isnan(np.sum(self.q3)), "NaN values in Q3 matrix"

        t, tau0, tau1 = [self.t, self.tau0, self.tau1]
        if 0 <= t <= tau0:
            return linalg.expm(np.array(self.q1*t))
        elif tau1 <= t <= tau0:
            return np.matmul(linalg.expm(np.array(self.q1*tau1)), linalg.expm(np.array(self.q2*(t-tau1))))
        elif tau0 < t:
            return np.matmul(np.matmul(linalg.expm(np.array(self.q1*tau1)), linalg.expm(np.array(self.q2*(tau0-tau1)))), linalg.expm(np.array(self.q3*(t-tau0))))
        else:
            return np.zeros(shape=(4,4))