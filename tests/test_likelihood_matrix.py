import numpy as np
import math
from dismal.likelihood_matrix import LikelihoodMatrix
from dismal.utils import expect_states1_2, expect_state_3

one_by_ten_S = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])


class TestLikelihoodMatrix:

    def test_alpha_matrix(self):
        """Check against hardcoded alpha matrix"""
        pass

    def test_beta_matrix(self):
        pass

    def test_gamma_matrix(self):
        pass

    def test_two_stage_model_conforms_to_theory1(self):

        theoretical_lm = np.array([
            [expect_states1_2(i, 1) for i in range(0, 10)],
            [expect_states1_2(i, 1) for i in range(0, 10)],
            [expect_state_3(i, 1, 2, 1) for i in range(0, 10)]])

        calculated_lm = LikelihoodMatrix(
            {"theta0": 1, "theta1": 1, "theta2": 1, "t0": 2}, S=one_by_ten_S)

        np.testing.assert_allclose(theoretical_lm[0], calculated_lm[0])
        np.testing.assert_allclose(theoretical_lm[1], calculated_lm[1])
        np.testing.assert_allclose(theoretical_lm[2], calculated_lm[2])

    def test_three_stage_model_conforms_to_theory1(self):

        theoretical_lm = np.array([
            [expect_states1_2(i, 1) for i in range(0, 10)],
            [expect_states1_2(i, 1) for i in range(0, 10)],
            [expect_state_3(i, 1, 2, 1) for i in range(0, 10)]])

        calculated_lm = LikelihoodMatrix(
            {"theta0": 1, "theta1": 1, "theta2": 1,
             "theta1_prime": 1, "theta2_prime": 1, "t1": 1, "v": 1}, S=one_by_ten_S)

        np.testing.assert_allclose(theoretical_lm[0], calculated_lm[0])
        np.testing.assert_allclose(theoretical_lm[1], calculated_lm[1])
        np.testing.assert_allclose(theoretical_lm[2], calculated_lm[2])

    # def test_three_stage_model_conforms_to_theory2(self):
    """Is the theory right here?"""

    #     theoretical_lm = np.array([
    #         [expect_states1_2(i, 5) for i in range(0, 10)],
    #         [expect_states1_2(i, 5) for i in range(0, 10)],
    #         [expect_state_3(k=i, theta=5, tau=10, a=5) for i in range(0, 10)]])

    #     params = {'theta0': 5,
    #               'theta1': 5,
    #               'theta2': 5,
    #               'theta1_prime': 5,
    #               'theta2_prime': 5,
    #               't1': 5,
    #               'v': 5}
    #     calculated_lm = LikelihoodMatrix(params, one_by_ten_S)

    #     np.testing.assert_allclose(theoretical_lm[0], calculated_lm[0])
    #     np.testing.assert_allclose(theoretical_lm[1], calculated_lm[1])
    #     np.testing.assert_allclose(theoretical_lm[2], calculated_lm[2])
