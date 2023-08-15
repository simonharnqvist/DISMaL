from dismal.markov_matrices import GeneratorMatrix
import numpy as np
import itertools
from scipy import linalg
import mpmath

popsizes = [0.01, 0.1, 1, 10, 100]
mig_rates = [0, 0.0001, 0.01, 1, 10]
popsize_permuts = [i for i in itertools.permutations(popsizes, 3)]
mig_rate_permuts = [i for i in itertools.permutations(mig_rates, 2)]

def test_q1_sums_to_zero():
     
     for popsize_permutation in popsize_permuts:
          for mig_rate_permutation in mig_rate_permuts:
              m1_prime_star, m2_prime_star = mig_rate_permutation[0], mig_rate_permutation[1]
              theta1, theta1_prime, theta2_prime = popsize_permutation[0], popsize_permutation[1], popsize_permutation[2]

              assert np.isclose(GeneratorMatrix(matrix_type="Q1",
                                                m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star, theta1=theta1,
                                                  theta1_prime=theta1_prime, theta2_prime=theta2_prime).sum(), 0)

def test_q2_sums_to_zero():
     
     for popsize_permutation in popsize_permuts:
          for mig_rate_permutation in mig_rate_permuts:
              m1_star, m2_star = mig_rate_permutation[0], mig_rate_permutation[1]
              theta1, theta2 = popsize_permutation[0], popsize_permutation[1]

              assert np.isclose(GeneratorMatrix(matrix_type="Q2",
                                                m1_star=m1_star, m2_star=m2_star,
                                                  theta1=theta1, theta2=theta2).sum(), 0)
              
            
def test_q1_correct_vals():
     q1 = GeneratorMatrix(matrix_type="Q1", theta1=1, theta1_prime=2, theta2_prime=3, m1_prime_star=1.5, m2_prime_star=2.5)
     correct_q1 = np.array([[-(1/2+3/4), 0, 3/4, 1/2],
                           [0, -(1/3+5/6), 5/6, 1/3],
                           [5/12, 3/8, -((0.75+(5/6))/2), 0],
                           [0, 0, 0, 0]])
     
     assert np.isclose(q1.matrix, correct_q1).all()

def test_q2_correct_vals():
     q2 = GeneratorMatrix(matrix_type="Q2", theta1=1, theta2=3, m1_star=1.5, m2_star=2.5)
     correct_q2 = np.array([[-(2.5), 0, 1.5, 1],
                           [0, -(1/3+5/6), 5/6, 1/3],
                           [5/12, 3/4, -((1.5+(5/6))/2), 0],
                           [0, 0, 0, 0]])
     
     assert np.isclose(q2.matrix, correct_q2).all()

def test_p_matrix_equivalent_to_expm():
     matrix = GeneratorMatrix(matrix_type="Q2", theta1=1, theta2=1, m1_star=2, m2_star=2)
     eigenvects, eigenvals = matrix.eigen()
     t = 1
     inv_eigenvects = linalg.inv(eigenvects)
     eigen_expm = inv_eigenvects @ np.diag(np.exp(eigenvals * t)) @ eigenvects

     matr = linalg.expm(matrix.matrix)
     linalg_expm = np.array(matr.tolist(), dtype=float)

     assert np.isclose(linalg_expm, eigen_expm).all()
