import itertools
import numpy as np
import math

def tally_to_counts(x):
    return list(itertools.chain(*[[i] * x[i] for i in range(0, len(x))]))


def expect_states1_2(k, theta):
    return -np.log((1/(theta+1)) * (theta/(theta+1))**k)

def expect_state_3(k, theta, tau, a):
    return -np.log((math.exp(-theta*tau) * 
            ((a*theta)**k) / ((1+a*theta)**(k+1))) \
    * np.sum([(((1/a+theta)**l) 
    * (tau**l))/math.factorial(l) for l in range(0,k+1)]))