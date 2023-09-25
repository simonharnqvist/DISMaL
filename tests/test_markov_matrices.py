from dismal.markov_matrices import TransitionRateMatrix, StochasticMatrix
import numpy as np
from scipy import linalg

def test_TransitionRateMatrix_correct_values_zero_migration():
    q_test = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]).matrix
    q_correct = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_almost_equal(q_test, q_correct)

def test_TransitionRateMatrix_correct_values_migration():
    q_test = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0.5, 0.2]).matrix
    q_correct = np.array([[-1.5, 0., 0.5, 1.], [0., -1.2, 0.2, 1.], [0.1, 0.25, -0.35, 0.], [0., 0., 0., 0.]])
    np.testing.assert_almost_equal(q_test, q_correct)

def test_TransitionRateMatrix_correct_eigenvalues_zero_migration():
    eigv_test = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]).eigen()[1]
    eigv_correct = np.array([-1., -1.,  0.,  0.])
    np.testing.assert_almost_equal(eigv_test, eigv_correct)

def test_StochasticMatrix_no_migration():
    Q = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0])
    P_test = StochasticMatrix(Q, 1).matrix
    P_correct = np.array([[0.36787944, 0.        , 0.        , 0.63212056],
                       [0.        , 0.36787944, 0.        , 0.63212056],
                       [0.        , 0.        , 1.        , 0.        ],
                       [0.        , 0.        , 0.        , 1.        ]])
    
    np.testing.assert_almost_equal(P_test, P_correct)

def test_StochasticMatrix_equiv_to_expm():
    Q = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0])
    P_test = StochasticMatrix(Q, 1).matrix
    P_expm = linalg.expm(Q.matrix)
    np.testing.assert_almost_equal(P_test, P_expm)

