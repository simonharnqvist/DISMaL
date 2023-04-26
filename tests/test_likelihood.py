from dismal.generator_matrices import GeneratorMatrix
from dismal.likelihood import p_matrix
from numpy import linalg
import numpy as np

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





