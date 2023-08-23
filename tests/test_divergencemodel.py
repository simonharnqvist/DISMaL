from dismal.divergencemodel import DivergenceModel
from dismal.markov_matrices import TransitionRateMatrix
import numpy as np
import math

def test_get_initial_values():
    mod = DivergenceModel()
    mod.add_epoch(index=0, deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(index=1, deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(index=2, deme_ids=["ancestral"], migration=False)

    s1, s2, s3 = [np.ones(10)]*3

    ivs = mod.get_initial_values(s1=s1, s2=s2, s3=s3, blocklen=100)
    np.testing.assert_almost_equal(ivs, [4.5, 4.5, 4.5, 4.5, 4.5, 0.045, 0.045, 0, 0, 0, 0])

def test_generate_markov_chain():
    mod = DivergenceModel()
    mod.add_epoch(index=0, deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(index=1, deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(index=2, deme_ids=["ancestral"], migration=False)

    param_vals = np.array([1,1,1,1,1,1,1,0,0,0,0])
    Qs = mod.generate_markov_chain(param_vals)
    Q = TransitionRateMatrix(thetas=[1,1], ms=[0,0]).matrix

    np.testing.assert_almost_equal(Qs[0].matrix, Q)
    np.testing.assert_almost_equal(Qs[1].matrix, Q)

def test_log_likelihood_from_params():
    mod = DivergenceModel()
    mod.add_epoch(index=0, deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(index=1, deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(index=2, deme_ids=["ancestral"], migration=False)
    param_vals = np.array([1,1,1,1,1,1,1,0,0,0,0])
    logll = mod._log_likelihood_from_params(param_vals, s1=np.ones(10), s2=np.ones(10), s3=np.ones(10))
    assert math.isclose(logll, 103.93615780591482)