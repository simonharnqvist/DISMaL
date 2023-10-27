import numpy as np
from scipy.stats import poisson
from dismal.markov_matrices import StochasticMatrix, TransitionRateMatrix
from collections import Counter
import math

    
def _poisson_cdf(s, t, eigenvalues):
    """Cumulative Poisson probability distribution of segregating sites.

    Args:
        s (ndarray): Segregating sites per locus in a given state.
        t (float): Epoch duration
        eigenvalues (ndarray): Eigenvalues of transition rate matrix of epoch.

    Returns:
        ndarray: Matrix n(eigenvalues) x n(s) of PoisCDF(s).
    """
    if t is None:
        return np.zeros(shape=(len(eigenvalues), len(s)))
    elif t == 0:
        return np.ones(shape=(len(eigenvalues), len(s)))
    else:
        s_counter = Counter(s)
        return np.concatenate(
            [np.transpose([
                poisson.cdf(s_val, t*(eigenvalues+1))] * s_count) 
                        for s_val, s_count in s_counter.items()], axis=1)
            


def _transform_eigenvalues_s(s, eigenvalues, start_time, end_time):
    """Transform s counts by eigenvalues to generate n(eigenvalues) x n(s) matrix"

    Args:
        s (ndarray): Segregating sites per locus in a given state.
        eigenvalues (ndarray): Eigenvalues of transition rate matrix of epoch.
        start_time (float): Epoch start time in Ne generations.
        end_time (float): Epoch end time in Ne generations.

    Returns:
        ndarray: Matrix n(eigenvalues) x n(s) of s-transformed eigenvalues.
    """

    s_counter = Counter(s)

    eigenvalues_s = np.concatenate([
        np.transpose([
            (eigenvalues/(eigenvalues+1)) 
            * (1/(eigenvalues+1)) ** s_val] 
            * s_count) 
            for s_val, s_count in s_counter.items()], axis=1)
    
    eigenvalues_exp = np.transpose(np.exp(eigenvalues * start_time))
    pois_start = _poisson_cdf(s, start_time, eigenvalues)
    pois_end = _poisson_cdf(s, end_time, eigenvalues)

    return np.transpose(np.transpose(eigenvalues_s) 
                        * eigenvalues_exp 
                        * np.transpose((pois_start - pois_end)))


def _state_log_likelihood(QQs, Ps, As, state_idx):
    return np.sum(
        np.log(QQs[0][state_idx, 0:-1]
               @ As[0]
               + Ps[0][state_idx, 0:-1]
               @ QQs[1][0:-1, 0:-1] 
               @ As[1]
               + (1 - (Ps[0][:]@Ps[1][:])[state_idx, -1])
               * As[2]))


def neg_logl(Qs, ts, s1, s2, s3):
    """Negative log-likelihood of parameter set given data.

    Args:
        Qs (iterable): TransitionRateMatrix objects, ordered from time 0 backwards (recent first).
        ts (iterable): Epoch durations, ordered from time 0 backwards (recent first).
        s1 (ndarray): Counts of segregating sites per locus in state 1.
        s2 (ndarray): Counts of segregating sites per locus in state 2.
        s3 (ndarray): Counts of segregating sites per locus in state 3.

    Returns:
        float: Negative log-likelihood of parameter set.
    """
    S = [s1, s2, s3]
    state_log_likelihoods = np.zeros(3)

    start_times = start_times = [0] + [sum(ts[0:i]) for i in range(1,len(ts)+1)]
    end_times = start_times[1:] + [None]
    
    QQs = [-Q.eigenvectors_inv @ np.diag(Q.eigenvectors[:, -1]) for Q in Qs[:-1]]
    Ps = [StochasticMatrix(Q, t=ts[idx]) for idx, Q in enumerate(Qs[:-1])]
    Q_eigvals = [np.array(-Q.eigenvalues[0:3]) for Q in Qs[:-1]]
    Q_eigvals.append(np.array([Qs[-1][0, 3]]))

    for state_idx in [0, 1, 2]:
        As = [_transform_eigenvalues_s(S[state_idx], 
                                      eigenvalues=Q_eigvals[epoch_idx],
                                      start_time=start_times[epoch_idx],
                                      end_time=end_times[epoch_idx])
                                      for epoch_idx in range(len(Qs))]

        state_log_likelihoods[state_idx] = (
            _state_log_likelihood(QQs, Ps, As, state_idx))
        
    return -np.sum(state_log_likelihoods)


def pr_s(s, state, thetas, epoch_durations, mig_rates):
    """Convenience function to calculate Pr(S=s) for a single value of s, given parameter set"""
    q1 = TransitionRateMatrix(single_deme=False,
                     thetas=[thetas[0], thetas[1]],
                     ms=[mig_rates[0], mig_rates[1]])
    q2 = TransitionRateMatrix(single_deme=False,
                     thetas=[thetas[2], thetas[3]],
                     ms=[mig_rates[2], mig_rates[3]])
    q3 = TransitionRateMatrix(single_deme=True,
                     thetas=[thetas[4]],
                     ms=[0])

    Qs = [q1, q2, q3]

    start_times = start_times = [0] + [sum(epoch_durations[0:i]) for i in range(1,len(epoch_durations)+1)]
    end_times = start_times[1:] + [None]  

    QQs = [-Q.eigenvectors_inv @ np.diag(Q.eigenvectors[:, -1]) for Q in Qs[:-1]]
    Ps = [StochasticMatrix(Q, t=epoch_durations[idx]) for idx, Q in enumerate(Qs[:-1])]
    Q_eigvals = [np.array(-Q.eigenvalues[0:3]) for Q in Qs[:-1]]
    Q_eigvals.append(np.array([Qs[-1][0, 3]]))

    As = [_transform_eigenvalues_s([s], 
                               eigenvalues=Q_eigvals[epoch_idx],
                               start_time=start_times[epoch_idx],
                               end_time=end_times[epoch_idx])
                                      for epoch_idx in range(len(Qs))]

    state_idx = state-1
    negll = _state_log_likelihood(QQs, Ps, As, state_idx)

    return math.exp(negll)
        




    

