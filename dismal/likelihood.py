import numpy as np
from scipy.stats import poisson
from dismal.markov_matrices import StochasticMatrix

def _epoch_durations(ts):
    durations = [ts[0]]
    for t_idx, t in enumerate(ts[1:]):
        durations.append(t - durations[t_idx-1])
    return durations

def _poisson_cdf(s, t, eigenvalues):
    if t is None:
        return np.zeros(shape=(len(eigenvalues), len(s)))
    elif t == 0:
        return np.ones(shape=(len(eigenvalues), len(s)))
    else:
        return np.transpose(np.array([poisson.cdf(s_val, t*(eigenvalues+1)) for s_val in s]))

def transform_eigenvalues_s(s, eigenvalues, start_time, end_time):
    """Transform s counts by eigenvalues to generate len(eigen) x len(s) matrix"""
    eigenvalues_s = (np.transpose([(eigenvalues/(eigenvalues+1)) 
                                  * (1/(eigenvalues+1)) ** s_val
                                  for s_val in s]))
    eigenvalues_exp = np.transpose(np.exp(eigenvalues * start_time))
    pois_start = _poisson_cdf(s, start_time, eigenvalues)
    pois_end = _poisson_cdf(s, end_time, eigenvalues)

    return np.transpose(np.transpose(eigenvalues_s) 
                        * eigenvalues_exp 
                        * np.transpose((pois_start - pois_end)))

def state_log_likelihood(QQs, Ps, As, state_idx):
    return np.sum(
        np.log(QQs[0][state_idx, 0:-1]
               @ As[0]
               + Ps[0][state_idx, 0:-1]
               @ QQs[1][0:-1, 0:-1] 
               @ As[1]
               + (1 - (Ps[0][:]@Ps[1][:])[state_idx, -1])
               * As[2]))


def log_likelihood(Qs, ts, s1, s2, s3):
    """Log-likelihood of parameter set given dataset."""
    S = [s1, s2, s3]
    state_log_likelihoods = np.zeros(3)
    
    epoch_dur = _epoch_durations(ts)
    start_times = [0] + ts
    end_times = ts + [None]

    QQs = [-Q.eigenvectors_inv @ np.diag(Q.eigenvectors[:, -1]) for Q in Qs[:-1]]
    Ps = [StochasticMatrix(Q, t=epoch_dur[idx]) for idx, Q in enumerate(Qs[:-1])]
    Q_eigvals = [np.array(-Q.eigenvalues[0:3]) for Q in Qs[:-1]]
    Q_eigvals.append(np.array([Qs[-1][0, 3]]))

    for state_idx in [0, 1, 2]:
        As = [transform_eigenvalues_s(S[state_idx], 
                                      eigenvalues=Q_eigvals[epoch_idx],
                                      start_time=start_times[epoch_idx],
                                      end_time=end_times[epoch_idx])
                                      for epoch_idx in range(len(Qs))]

        state_log_likelihoods[state_idx] = (
            state_log_likelihood(QQs, Ps, As, state_idx))
        
    return -np.sum(state_log_likelihoods)
        




    

