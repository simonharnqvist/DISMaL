import numpy as np
from dismal.demography import Epoch
from dismal.likelihood_matrix import LikelihoodMatrix

def test_two_stage_ll_matrix_no_mig():
    params = [1,1,1,1,0,0]
    S = np.ones(shape=(3, 10))
    epochs = [
        Epoch(id=0, allow_migration=False),
        Epoch(id=1, allow_migration=True)
    ]

    lm = LikelihoodMatrix(params, S, epochs).matrix
    correct_lm = np.array([[0.69314718, 1.38629436, 2.07944154, 2.77258872, 3.4657359,
        4.15888308, 4.85203026, 5.54517744, 6.23832463, 6.93147181],
       [0.69314718, 1.38629436, 2.07944154, 2.77258872, 3.4657359 ,
        4.15888308, 4.85203026, 5.54517744, 6.23832463, 6.93147181],
       [1.69314718, 1.28768207, 1.47000363, 1.92676203, 2.51982575,
        3.1755854 , 3.85657438, 4.54627477, 5.2385621 , 5.9315183 ]])

    np.testing.assert_array_almost_equal(lm, correct_lm)

def test_three_stage_ll_matrix_no_mig():
    epochs = [Epoch(id=0, allow_migration=False),
              Epoch(id=1, allow_migration=True),
              Epoch(id=2, allow_migration=True)]

    S = np.ones(shape=(3, 10))
    params=[1,1,1,1,1,1,1,0,0,0,0]

    lm = LikelihoodMatrix(params=params, S=S, epoch_objects=epochs).matrix
    correct_lm = np.array([[0.78257772, 1.68287544, 2.63760526, 3.55304993, 4.37912461,
        5.13081962, 5.8442701 , 6.54329475, 7.23791671, 7.93139191],
       [0.78257772, 1.68287544, 2.63760526, 3.55304993, 4.37912461,
        5.13081962, 5.8442701 , 6.54329475, 7.23791671, 7.93139191],
       [1.69314718, 1.28768207, 1.47000363, 1.92676203, 2.51982575,
        3.1755854 , 3.85657438, 4.54627477, 5.2385621 , 5.9315183 ]])
    
    np.testing.assert_array_almost_equal(lm, correct_lm)

def test_three_stage_ll_matrix_with_mig():
    epochs = [Epoch(id=0, allow_migration=False),
              Epoch(id=1, allow_migration=True),
              Epoch(id=2, allow_migration=True)]

    S = np.ones(shape=(3, 10))
    params=[1,1,1,1,1,1,1,0.5,0.5,0.5,0.5]

    lm = LikelihoodMatrix(params=params, S=S, epoch_objects=epochs).matrix
    correct_lm = np.array([[0.84242651, 1.60213032, 2.31050187, 3.00141071, 3.69062159,
        4.38159278, 5.07389379, 5.76677652, 6.45985363, 7.15298459],
       [0.84242651, 1.60213032, 2.31050187, 3.00141071, 3.69062159,
        4.38159278, 5.07389379, 5.76677652, 6.45985363, 7.15298459],
       [1.61515115, 1.48896043, 1.78984599, 2.29930387, 2.91485817,
        3.57928333, 4.26319223, 4.95374699, 5.64625299, 6.33925875]])
    
    np.testing.assert_array_almost_equal(lm, correct_lm)

    
    