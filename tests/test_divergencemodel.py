from dismal.divergencemodel import DivergenceModel
from dismal.markov_matrices import TransitionRateMatrix
import numpy as np
import math

def test_generate_markov_chain():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["ancestral"], migration=False)

    param_vals = np.array([1,1,1,1,1,1,1,0,0,0,0])
    Qs = mod._generate_markov_chain(param_vals)
    Q = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]).matrix

    np.testing.assert_almost_equal(Qs[0].matrix, Q)
    np.testing.assert_almost_equal(Qs[1].matrix, Q)

def test_generate_markov_chain_directional_migration():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True, asymmetric_migration=True, migration_direction=("pop2", "pop1"))
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True, asymmetric_migration=True, migration_direction=("pop1", "pop2"))
    mod.add_epoch(deme_ids=["ancestral"], migration=False)

    param_vals = np.array([1, 1, 1, 1, 1, 1, 1, 0.5, 0.25])
    Qs = mod._generate_markov_chain(param_vals)
    
    Q1 = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0, 0.5]).matrix
    Q2 = TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0.25, 0]).matrix

    np.testing.assert_almost_equal(Qs[0].matrix, Q1)
    np.testing.assert_almost_equal(Qs[1].matrix, Q2)

def test_log_likelihood_from_params():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["ancestral"], migration=False)
    param_vals = np.array([1,1,1,1,1,1,1,0,0,0,0])

    r_validated_logll = 103.9362

    logll = mod.neg_log_likelihood(param_vals, 
                                            s1=np.array([i for i in range(10)]), 
                                            s2=np.array([i for i in range(10)]), 
                                            s3=np.array([i for i in range(10)]))
    
    assert math.isclose(logll, 103.93615780591482)

def test_from_dict_spec():
    dict_spec = {"epochs": 3, 
                 "deme_ids": [("pop1", "pop2"), ("pop1", "pop2"), ("ancestral", )],
                              "migration": (True, True, False), 
                              "asym_migration": (True, True, False), 
                              "migration_direction": (("pop1", "pop2"), ("pop1", "pop2"))}
    mod = DivergenceModel.from_dict_spec(dict_spec)

    assert len(mod.epochs) == 3
    assert mod.n_theta_params == 5
    assert mod.n_t_params == 2
    assert mod.n_m_params == 2

    assert mod.epochs[0].migration is True
    assert mod.epochs[1].migration is True
    assert mod.epochs[2].migration is False
    assert mod.epochs[0].asymmetric_migration is True
    assert mod.epochs[1].asymmetric_migration is True
    assert mod.epochs[2].asymmetric_migration is False
    assert mod.epochs[0].migration_direction == ("pop1", "pop2")
    assert mod.epochs[1].migration_direction == ("pop1", "pop2")
    assert mod.epochs[0].deme_ids == ("pop1", "pop2")
    assert mod.epochs[1].deme_ids == ("pop1", "pop2")
    assert mod.epochs[2].deme_ids == ("ancestral", )

def test_model_fitting():
    dict_spec = {"epochs": 3, 
                 "deme_ids": [("pop1", "pop2"), ("pop1", "pop2"), ("ancestral", )],
                              "migration": (True, True, False), 
                              "asym_migration": (True, True, False), 
                              "migration_direction": (("pop1", "pop2"), ("pop1", "pop2"))}
    mod = DivergenceModel.from_dict_spec(dict_spec)
    s1, s2, s3 = [np.ones(10)]*3
    mod.fit(s1, s2, s3, blocklen=500)

    assert len(mod.inferred_params) == 9