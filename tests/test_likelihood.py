from dismal.generator_matrices import GeneratorMatrix
from dismal.likelihood import p_matrix, likelihood_matrix
from numpy import linalg
import numpy as np
from scipy.stats import poisson
import math

def test_p_matrix_is_identity_matrix_if_t_is_zero():
    """Irrespective of parameter values, P matrix should be identity matrix at time 0"""
    
    parameter_sets = [[5,5,5,5,5,5,5,0,0,0,0],
                      [5,5,5,5,5,5,5,5,5,5,5],
                      [0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0]]
    
    for params in parameter_sets:
        q1, q2, q3 = GeneratorMatrix.from_params(params)
        g, eigenvalues = q1.eigen()
        ginv = linalg.inv(g)
        t = 0
        assert np.allclose(np.identity(4), p_matrix(g, ginv, eigenvalues, t))

s_vals = [i for i in range(0, 10)]
q1, q2, q3 = GeneratorMatrix.from_params([1,1,1,1,1,1,1,0,0,0,0])
likelihood_mat = np.exp(-np.array(likelihood_matrix(q1, q2, q3, 1, 1, s_vals=s_vals)))

def test_likelihood_matrix_matches_theory_state1_2():
    def expect_states1_2(k, theta):
        return (1/(theta+1)) * (theta/(theta+1))**k
    
    expected = [expect_states1_2(k, 1) for k in s_vals]
    observed1 = likelihood_mat[0][0:len(s_vals)]
    observed2 = likelihood_mat[1][0:len(s_vals)]
    assert np.allclose(observed1, observed2)
    assert np.allclose(expected, observed1)


def test_likelihood_matrix_matches_theory_state3():
    
    def expect_state3_cwh(k, theta, tau, a):
        return (math.exp(-theta*tau) * ((a*theta)**k) / ((1+a*theta)**(k+1))) * np.sum([(((1/a+theta)**l) * (tau**l))/math.factorial(l) for l in range(0,k+1)])
    
    expected = [expect_state3_cwh(k, 1, 2, 1) for k in s_vals]
    observed = likelihood_mat[2][0:len(s_vals)]
    assert np.allclose(expected, observed)
    






