from dismal.likelihood import transform_eigenvalues_s, log_likelihood, state_log_likelihood
from dismal.divergencemodel import DivergenceModel
from dismal.markov_matrices import TransitionRateMatrix, StochasticMatrix
import numpy as np
import math

def test_transform_eigenvalues_s():

    r_validated_array = np.array([[4.32332358e-01, 1.48498538e-01, 4.04154480e-02, 8.92978372e-03,
                                   1.64540679e-03, 2.58806383e-04, 3.54203557e-05],
                                   [4.32332358e-01, 1.48498538e-01, 4.04154480e-02, 8.92978372e-03,
                                    1.64540679e-03, 2.58806383e-04, 3.54203557e-05],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    np.testing.assert_allclose(
        transform_eigenvalues_s(s=np.array([0, 1, 2, 3, 4, 5, 6]), 
                                eigenvalues=np.array([1,1,0]), 
                                start_time=0, 
                                end_time=1), 
        r_validated_array)
    
def test_state_log_likelihood():

    s = np.concatenate([[0]*1000, [1]*1000, [2]*1000])
    eigenvalues = np.array([1,1,0])
    
    Qs = [TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]),
      TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]),
      TransitionRateMatrix(single_deme=False, thetas=[1], ms=[0])]
    
    epoch_dur = [1,1]

    Ps = [StochasticMatrix(Q, t=epoch_dur[idx]) for idx, Q in enumerate(Qs[:-1])]
    QQs = [-Q.eigenvectors_inv @ np.diag(Q.eigenvectors[:, -1]) for Q in Qs[:-1]]
    Q_eigvals = [np.array(-Q.eigenvalues[0:3]) for Q in Qs[:-1]]
    Q_eigvals.append(np.array([Qs[-1][0, 3]]))

    start_times = [0, 1, 2]
    end_times = [1, 2, None]
    As = [transform_eigenvalues_s(s, 
                                  Q_eigvals[idx], 
                                  start_time=start_times[idx], 
                                  end_time=end_times[idx]) for idx in range(0,3)]
    
    state_lls = [state_log_likelihood(QQs, Ps, As, i) for i in range(0, 3)]
    r_validated_state_lls = [-4158.883, -4158.883, -5984.496]

    np.testing.assert_allclose(state_lls, r_validated_state_lls, atol=10-3)


def test_log_likelihood():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=False)
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=False)
    mod.add_epoch(deme_ids=["ancestral"], migration=False)
    Qs = mod.generate_markov_chain(param_vals=np.array([1,1,1,1,1,1,1,0,0,0,0]))
    logl = log_likelihood(Qs, ts=[1,2], s1 = np.array([i for i in range(10)]), 
                          s2=np.array([i for i in range(10)]), 
                          s3=np.array([i for i in range(10)]))
    assert math.isclose(logl, 103.93615780591482)

def test_log_likelihood_with_mig():
    mod = DivergenceModel()
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["pop1", "pop2"], migration=True)
    mod.add_epoch(deme_ids=["ancestral"], migration=False)
    Qs = mod.generate_markov_chain(param_vals=np.array([1,1,1,1,1,1,1,1,1,1,1]))
    logl = log_likelihood(Qs, ts=[1,2], s1 = np.array([i for i in range(10)]), 
                          s2=np.array([i for i in range(10)]), 
                          s3=np.array([i for i in range(10)]))
    assert math.isclose(logl, 95.92016, abs_tol=1e-5)

def test_log_likelihood_with_simple_large_number_no_mig():
    s1 = np.concatenate([[0]*1000, [1]*1000, [2]*1000])
    s2 = np.concatenate([[0]*1000, [1]*1000, [2]*1000])
    s3 = np.concatenate([[0]*1000, [1]*1000, [2]*1000])

    Qs = [
        TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]),
        TransitionRateMatrix(single_deme=False, thetas=[1,1], ms=[0,0]),
        TransitionRateMatrix(single_deme=True, thetas=[1], ms=[0])]
    
    ts = [1,2]

    logl = log_likelihood(Qs, ts, s1, s2, s3)
    r_validated_logl = 14302.26

    assert math.isclose(logl, r_validated_logl, abs_tol=10-3)

def test_log_likelihood_with_simulated_data_no_mig():
    s1 = np.concatenate([
        [0] * int(1.6615e+04), 
        [1] * int(2.8380e+03), 
        [2] * int(4.6100e+02), 
        [3] * int(7.0000e+01), 
        [4] * int(1.4000e+01),
        [5] * int(2.0000e+00)])
    
    s2 = np.concatenate([
        [0] * int(1.6728e+04), 
                   [1] * int(2.7590e+03), 
                   [2] * int(4.2600e+02), 
                   [3] * int(7.2000e+01), 
                   [4] * int(1.3000e+01),
                   [5] * int(2.0000e+00)])

    s3 = np.concatenate([
        [0] * int(1.6698e+04), 
                   [1] * int(2.7480e+03), 
                   [2] * int(4.6100e+02), 
                   [3] * int(6.5000e+01), 
                   [4] * int(2.3000e+01),
                   [5] * int(5.0000e+00)])

    Qs = [
        TransitionRateMatrix(single_deme=False, thetas=[3,2], ms=[0,0]),
        TransitionRateMatrix(single_deme=False, thetas=[4,3], ms=[0,0]),
        TransitionRateMatrix(single_deme=True, thetas=[6], ms=[0])]
    
    ts = [4,8]

    logl = log_likelihood(Qs, ts, s1, s2, s3)

    r_validated_logll = 243270

    assert math.isclose(logl, r_validated_logll, abs_tol=1e-1)