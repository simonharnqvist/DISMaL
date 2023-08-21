import numpy as np
import scipy
from demography import Epoch

np.set_printoptions(suppress=True)

def log_likelihood(params, S, epochs, verbose=False):
    """Log likelihood evaluation for parameter set given data S=[s1, s2, s3...],
    with parameters given in order [thetas, taus, ms] starting with the MOST recent parameters, i.e. params[0] is the size of current population 0"""

    if np.isnan(params).any():
        return np.nan
        
    assert isinstance(epochs[0], Epoch)

    n_thetas = (len(epochs)-1)*2 + 1
    thetas = params[0:n_thetas]
    taus = params[n_thetas:n_thetas+len(epochs)]
    ms = params[n_thetas+len(epochs):]

    for epoch_idx, epoch in epochs:
        if epoch.allow_migration is True:
            

