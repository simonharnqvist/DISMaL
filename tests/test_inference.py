import numpy as np
from dismal.inference import DemographicModel

S = np.array([[1] * 10] * 3)

gim = DemographicModel(migration=True,
                       mig_rate_change=True,
                       asymmetric_m_star=True,
                       asymmetric_m_prime_star=True,
                       pop_size_change=True)

param_names = ["theta0", "theta1", "theta2",
               "theta1_prime", "theta2_prime", "t1", "v",
               "M1_star", "M2_star", "M1_prime_star", "M2_prime_star"]


def test_likelihood_scenario1():
    params = [3, 2, 4, 3, 6, 4, 4, 0.2, 0.4, 0.04, 0.08]
    ll = gim.composite_likelihood(params, param_names, S=S)
    rgim_ll = 87.0
    assert abs(ll-rgim_ll) < 0.1


def test_likelihood_scenario2():
    params = [3, 2, 4, 3, 6, 4, 4, 0.04, 0.08, 0.2, 0.4]
    ll = gim.composite_likelihood(params, param_names, S=S)
    rgim_ll = 83.8
    assert abs(ll-rgim_ll) < 0.1


def test_likelihood_scenario3():
    params = [3, 2, 4, 3, 6, 4, 4, 0.2, 0.4, 0, 0]
    ll = gim.composite_likelihood(params, param_names, S=S)
    rgim_ll = 90.6
    assert abs(ll-rgim_ll) < 0.1


def test_likelihood_scenario4():
    params = [3, 2, 4, 3, 6, 4, 4, 0, 0, 0.2, 0.4]
    ll = gim.composite_likelihood(params, param_names, S=S)
    rgim_ll = 84.0
    assert abs(ll-rgim_ll) < 0.1


def test_likelihood_scenario5():
    params = [3, 2, 4, 3, 6, 4, 4, 0, 0, 0, 0]
    ll = gim.composite_likelihood(params, param_names, S=S)
    rgim_ll = 96.1
    assert abs(ll-rgim_ll) < 0.1
