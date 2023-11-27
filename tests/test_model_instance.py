import numpy as np
from dismal.model_instance import ModelInstance
from dismal.markov_matrices import TransitionRateMatrix
from dismal.demography import Epoch
import math


def test_generate_markov_chain():
    epochs = [Epoch(n_demes=2, deme_ids=["pop1", "pop2"], migration=True),
              Epoch(n_demes=2, deme_ids=["pop1", "pop2"], migration=True),
              Epoch(n_demes=1, deme_ids=["ancestral"], migration=False)]
    
    param_vals = np.array([1,1,1,1,1,1,1,0,0,0,0])
    mod = ModelInstance(param_vals, epochs)
    mod._generate_markov_chain()
    Q = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]).matrix

    np.testing.assert_almost_equal(mod.Qs[0].matrix, Q)
    np.testing.assert_almost_equal(mod.Qs[1].matrix, Q)


def test_generate_markov_chain_directional_migration():
    epochs = [
        Epoch(n_demes=2, deme_ids=["pop1", "pop2"], migration=True, asymmetric_migration=True, migration_direction=("pop2", "pop1")),
        Epoch(n_demes=2, deme_ids=["pop1", "pop2"], migration=True, asymmetric_migration=True, migration_direction=("pop1", "pop2")),
        Epoch(n_demes=1, deme_ids=["ancestral"], migration=False)]

    param_vals = np.array([1, 1, 1, 1, 1, 1, 1, 0.5, 0.25])
    mod = ModelInstance(param_vals, epochs)
    mod._generate_markov_chain()
    
    Q1 = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0, 0.5]).matrix
    Q2 = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0.25, 0]).matrix

    np.testing.assert_almost_equal(mod.Qs[0].matrix, Q1)
    np.testing.assert_almost_equal(mod.Qs[1].matrix, Q2)


def test_transform_eigenvalues_s():

    r_validated_array = np.array([[4.32332358e-01, 1.48498538e-01, 4.04154480e-02, 8.92978372e-03,
                                   1.64540679e-03, 2.58806383e-04, 3.54203557e-05],
                                   [4.32332358e-01, 1.48498538e-01, 4.04154480e-02, 8.92978372e-03,
                                    1.64540679e-03, 2.58806383e-04, 3.54203557e-05],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    np.testing.assert_allclose(
        ModelInstance._transform_eigenvalues_s(s_max=6, 
                                eigenvalues=np.array([1,1,0]), 
                                start_time=0, 
                                end_time=1), r_validated_array)
    

def test_pr_s_state1():

    epochs = [
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=1, deme_ids=("pop1",), migration=False)]
    
    mod = ModelInstance([1,1,1,1,1,1,1], epochs)

    theoretical_expectation = np.array([0.5, 0.25, 0.125, 0.0625])
    pr_s_result = mod.pr_s(s_max=3, state=1, neglog=False)

    np.testing.assert_allclose(theoretical_expectation, pr_s_result)


def test_pr_s_state3():

    epochs = [
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=1, deme_ids=("pop1",), migration=False)]
    
    mod = ModelInstance([1,1,1,1,1,1,1], epochs)

    rgim_validated = np.array([0.06766764, 0.1691691])
    pr_s_result = mod.pr_s(s_max=1, state=3, neglog=False)

    np.testing.assert_allclose(rgim_validated, pr_s_result)


def test_log_likelihood():
    
    s1, s2, s3 = np.array([1000]), np.array([1000]), np.array([1000])
       
    epochs = [
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=1, deme_ids=("pop1",), migration=False)]
    
    mod = ModelInstance([1,1,1,1,1,1,1], epochs)

    logl = mod.neg_composite_log_likelihood(s1, s2, s3)
    assert math.isclose(logl, 4079.442, abs_tol=1e-3)


def test_log_likelihood_with_mig():
    epochs = [
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=True),
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=True),
        Epoch(n_demes=1, deme_ids=("pop1",), migration=False)]
    
    mod = ModelInstance([1,1,1,1,1,1,1,1,1,1,1], epochs)

    logl = mod.neg_composite_log_likelihood(
        s1 = np.array([1]*10), 
        s2 = np.array([1]*10), 
        s3 = np.array([1]*10))
    assert math.isclose(logl, 95.92016, abs_tol=1e-5)


def test_log_likelihood_with_simple_large_number_no_mig():
    epochs = [
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=2, deme_ids=("pop1", "pop2"), migration=False),
        Epoch(n_demes=1, deme_ids=("pop1",), migration=False)]
    
    mod = ModelInstance([1,1,1,1,1,1,1], epochs)
    
    s1 = np.array([1000, 1000, 1000])
    s2 = np.array([1000, 1000, 1000])
    s3 = np.array([1000, 1000, 1000])

    logl = mod.neg_composite_log_likelihood(s1, s2, s3)
    r_validated_logl = 14302.26

    assert math.isclose(logl, r_validated_logl, abs_tol=10-3)

