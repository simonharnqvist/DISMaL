from dismal.likelihood import eigenvalue_transform, log_likelihood
from dismal.divergencemodel import DivergenceModel
import numpy as np
import math

def test_eigenvalue_transform():
    transformed = eigenvalue_transform(lmbdas=np.array([1,1,0]), s=np.ones(10), start_time=1, end_time=2)
    assert transformed.shape == (3,10) and math.isclose(transformed[0,0], 0.1590461864017892)

def test_log_likelihood():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=False)
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=False)
    mod.add_epoch(deme_ids=["ancestral"], migration=False)
    Qs = mod.generate_markov_chain(param_vals=np.array([1,1,1,1,1,1,1,0,0,0,0]))
    logl = log_likelihood(Qs, ts=[1,1], s1 = np.ones(10), s2=np.ones(10), s3=np.ones(10))
    assert math.isclose(logl, 103.93615780591482)

def test_log_likelihood_with_mig():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["ancestral"], migration=False)
    Qs = mod.generate_markov_chain(param_vals=np.array([1,1,1,1,1,1,1,1,1,1,1]))
    logl = log_likelihood(Qs, ts=[1,1], s1 = np.ones(10), s2=np.ones(10), s3=np.ones(10))
    assert math.isclose(logl, 95.92016, abs_tol=1e-5)