import numpy as np
from scipy.stats import poisson
from dismal.markov_matrices import StochasticMatrix

def _poisson_cdf(s_val, t, eigenvalues):
    if t is None:
        return 0
    elif t == 0:
        return 1 # same result but shortcut
    else:
        return poisson.cdf(s_val, t*(eigenvalues+1))

def eigenvalue_transform(lmbdas, s, start_time, end_time):
    assert isinstance(lmbdas, np.ndarray), "lambdas must be np array"
    
    return np.transpose(
                np.array(
                [lmbdas/(lmbdas+1)*(1/(lmbdas+1)) 
                 ** (idx) * np.exp(lmbdas * start_time) *(_poisson_cdf(idx, start_time, lmbdas)
                           -_poisson_cdf(idx, end_time, lmbdas)) 
                 for idx in range(0, len(s))])) * s
            
def state_3epoch_log_likelihood(QQs, As, Ps, state_idx):
    
    return np.sum(np.log(
        QQs[0][state_idx, 0:3] @ As[state_idx][0]
        + Ps[0][state_idx, 0:3] @ QQs[1][0:3, 0:3] @ As[state_idx][1]
        + (1 - ((Ps[0].matrix @ Ps[1].matrix)[state_idx, -1])) * As[state_idx][2]))

def log_likelihood(Qs, ts, s1, s2, s3):
    """
    ts = epoch duration vector
    """

    Ps = [StochasticMatrix(Q, t=ts[idx]) for idx, Q in enumerate(Qs[:-1])]
    QQs = [-Q.eigenvectors_inv @ np.diag(Q.eigenvectors[:, -1]) for Q in Qs[:-1]]
    Q_eigvals = [np.array(-Q.eigenvalues[0:3]) for Q in Qs[:-1]]
    Q_eigvals.append(np.array([Qs[-1][0,3]]))

    # convert epoch durations to start and end times
    start_times = [0]
    end_times = []
    for idx, time in enumerate(ts):
        start_times.append(sum(start_times) + time)
    end_times = start_times[1:] + [None]

    S = [s1, s2, s3]
    As = [[eigenvalue_transform(lmbdas=Q_eigvals[epoch_idx], s=s,
                                start_time=start_times[epoch_idx],
                                end_time=end_times[epoch_idx]) for epoch_idx, Q in enumerate(Qs)] 
                                for state_idx, s in enumerate(S)]
         
    log_likelihoods = np.zeros(shape=3)
    for state_idx in range(3):
        log_likelihoods[state_idx] = state_3epoch_log_likelihood(QQs, As, Ps, state_idx)

    logl = -np.sum(log_likelihoods)

    return logl
         




    

