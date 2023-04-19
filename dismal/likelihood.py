import math
from math import factorial
from mpmath import exp
import numpy as np
from utils import _opt_to_model_params, _model_to_opt_params
import matrices
import itertools
import scipy

def _expected_gs_u(s, theta, lmbda):
    """ Calculate the expected number of mutations for a fixed time span.

    Keyword arguments:
    s -- number of mutations
    theta -- effective population size 4*Ne*mu
    lmbda -- rate parameter of Poisson distribution
    """
    
    return (lmbda * (theta**s)) / ((lmbda+theta)**(s+1))

def _lambda_summation(l, theta, lmbda, t):
    """ Perform the series summation of equation 10 in Costa & Wilkinson-Herbots (2021).
    
    Keyword arguments:
    l -- number of mutations
    theta -- effective population size 4*Ne*mu
    lmbda -- rate parameter of Poisson distribution
    t -- time span in coalescent units
    """
    assert l < 100, f"Your s-values are too high (s={l} detected). Consider using smaller blocks."
    return sum([(((lmbda+theta)**s) * t**s) / factorial(s) for s in range(0, l+1)])

def _pdf_gs(s, theta, lmbda, t=None):
    """ Calculate the probability of seeing s mutations in a given time span.

    Keyword arguments:
    s -- number of mutations
    theta -- effective population size 4*Ne*mu
    lmbda -- rate parameter of Poisson distribution
    t -- time span in coalescent units
    """

    s = int(s)
    if t is None:
        gs = _expected_gs_u(s=s, theta=theta, lmbda=lmbda)
    else:
        gs = exp(-theta*t) * _expected_gs_u(s=s, theta=theta, lmbda=lmbda) * _lambda_summation(l=s, theta=theta, lmbda=lmbda, t=t)
    return gs


def _pdf_s(s, a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta, state):
    """ Probability density function of s, the number of nucleotide differences between two blocks.

    Keyword arguments:
    s -- number of nucleotide differences
    a -- relative size of ancestral population (relative to size of population 1 between tau0 and tau1)
    b -- relative size of the second population during period tau0 to tau1
    c1 -- relative size of the first population during period tau1 to present
    c2 -- relative size of the second population during period tau1 to present
    m1 -- migration rate from population 1 to population 2 during period tau0 to tau1
    m2 -- migration rate from population 2 to population 1 during period tau0 to tau1
    m1_prime -- migration rate from population 1 to population 2 during period tau1 to present
    m2_prime -- migration rate from population 2 to population 1 during period tau1 to present
    theta -- effective population size 4*Ne*mu
    state -- sampling state {1,2,3} where 1 means that both blocks are sampled from population 1, 2 means that both blocks are sampled from population 2, and 3 means one block per population
    """

    i = state-1
    s = int(s)

    if m1 is None:
        m1 = 0
    if m2 is None:
        m2 = 0
    if m1_prime is None:
        m1_prime = 0
    if m2_prime is None:
        m2_prime = 0
    assert not None in [s, a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta, state]

    q1 = matrices.GeneratorMatrix(m1_prime=m1_prime, m2_prime=m2_prime, c1=c1, c2=c2, matrix_type="Q1")
    q2 = matrices.GeneratorMatrix(m1=m1, m2=m2, b=b, matrix_type="Q2")
    q3 = matrices.GeneratorMatrix(a=a, matrix_type="Q3")
    g, alpha = q1.left_eigenvectors()
    c, beta = q2.left_eigenvectors()
    ginv = matrices.GeneratorMatrix.inverse(g)
    cinv = matrices.GeneratorMatrix.inverse(c)
    p1 = matrices.TransitionMatrix(q1, q2, q3, t=tau1, tau1=tau1, tau0=tau0)
    p2 = matrices.TransitionMatrix(q1, q2, q3, t=(tau0-tau1), tau1=tau1, tau0=tau0)

    # Lists of expectations; eigenvalues are rate parameters of nt difference distribution
    w_expect = np.array([_pdf_gs(s=s, theta=theta, lmbda=eigval) for eigval in alpha])
    tau1_w_expect = np.array([_pdf_gs(s=s, theta=theta, lmbda=eigval, t=tau1) for eigval in alpha])
    tau1_y_expect = np.array([_pdf_gs(s=s, theta=theta, lmbda=eigval, t=tau1) for eigval in beta])
    tau0_y_expect = np.array([_pdf_gs(s=s, theta=theta, lmbda=eigval, t=tau0) for eigval in beta])
    tau0_x_expect = np.array(_pdf_gs(s=s, theta=theta, lmbda=1/a, t=tau0))

    term1 = np.sum([float(ginv[i,k]) * float(g[k,3]) * (w_expect[k] - tau1_w_expect[k] *
                                          exp(-alpha[k]*tau1)) for k in range(0,4) if alpha[k] > 0])

    term2 = np.sum([[float(p1[i,j]) * float(cinv[j,k]) * float(c[k,3]) * (tau1_y_expect[k] - tau0_y_expect[k] *
                                            exp(-beta[k]*(tau0-tau1))) for k in range(0,4) if beta[k] > 0]
                   for j in range(0,3)])

    term3 = np.sum([[float(p1[i,j]) * float(p2[j,l]) * tau0_x_expect for l in range(0,3)] for j in range(0,3)])

    pr_s = float(-term1-term2+term3)

    assert 0 <= pr_s <= 1, f"Probabilities outside range [0,1]: p={pr_s}, term1={term1}, term2={term2}, term3={term3} for parameters {[s, a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta, state]}"

    return pr_s


def _sval_likelihood(s_val, s_count, params, state):
    """ Calculate the likelihood of a given value of s.

    Keyword arguments:
    s_val -- the number of nucleotide differences (the likelihood of which is calculated)
    s_counts -- the count of that value of s (i.e. how many times does that value of s occur in that state)
    params -- dictionary of model parameters
    state -- sampling state {1,2,3} where 1 means that both blocks are sampled from population 1, 2 means that both blocks are sampled from population 2, and 3 means one block per population
    """

    assert isinstance(params, dict), f"{params}"

    a = params["a"]
    b = params["b"]
    c1 = params["c1"]
    c2 = params["c2"]
    tau1 = params["tau1"]
    tau0 = params["tau0"]
    theta = params["theta"]

    if "m1" in list(params.keys()):
        m1 = params["m1"]
    else:
        m1 = 0
    if "m2" in list(params.keys()):
        m2 = params["m2"]
    else:
        m2 = 0
    if "m1_prime" in list(params.keys()):
        m1_prime = params["m1_prime"]
    else:
        m1_prime = 0
    if "m2_prime" in list(params.keys()):
        m2_prime = params["m2_prime"]
    else:
        m2_prime = 0


    p_s = _pdf_s(a=a, b=b, c1=c1, c2=c2, tau0=tau0, tau1 = tau1, m1=m1, m2=m2, m1_prime=m1_prime, m2_prime=m2_prime, theta=theta, state=state, s=s_val)
    assert 0 <= p_s <= 1, f"Probabilities outside range [0,1]: p={p_s} for parameters {params}"

    logL = np.log(p_s) * s_count
    assert logL < 0, "Positive log-likelihood detected - (positive) log-likelihoods must be negative"
    return logL


def _composite_neg_ll(params, X, parameter_names, verbose=True):
    """ Calculate the composite negative log likelihood of a parameter set, given dataset X.

    Keyword arguments:
    params -- optimisation parameters [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]
    X -- list of three dictionaries, each containing key-value pairs where the key is s, the number of differences, and the value is the corresponding count
    verbose -- whether to output parameter values and log-likelihoods during fitting
    """
    # Check for NaNs in input
    params = [0 if math.isnan(float(param)) else float(param) for param in params]

    param_dict = dict(zip(parameter_names, params))

    # Convert params for likelihood function
    model_params = _opt_to_model_params(param_dict)
    assert not any([np.isnan(param) for param in list(model_params.values())]), f"NaN values in converted model params {model_params}; original params {params}"

    assert isinstance(model_params, dict)
    # Multiply each s by likelihood of that s, and sum to generate composite LL
    log_likelihoods = list(itertools.chain(*[[_sval_likelihood(s_val = s, s_count=X[state-1][s], params = model_params, state=state) for s in X[state-1].keys()] 
                                         for state in [1,2,3]]))
    assert all(i < 0 for i in log_likelihoods), f"Positive log-likelihood detected - (positive) log-likelihoods must be negative: {log_likelihoods}"

    negll = -np.sum(log_likelihoods)

    if verbose:
        print(f"Parameters: {model_params}, -lnL: {negll}")

    assert negll > 0, "Negative -negll detected; negative log-likelihoods must be positive"

    return negll

def _optimise_negll(X, initial_vals, lower_bounds, optimisation_algo, verbose):
    """ Optimise negative log likelihood.
        
    Keyword arguments:
    X -- list of dictionaries of s values
    initial_vals -- dict of initial values in model parameters
    lower_bounds -- dict of lower bounds in model parameters
    optimisation_algo -- which algo to use (see scipy.optimize.minimize)
    """

    # Convert to optimisation parameters
    opt_iv = _model_to_opt_params(initial_vals)
    opt_lb = _model_to_opt_params(lower_bounds)

    parameter_names = list(opt_iv.keys())
    upper_bounds = [None] * len(parameter_names)
    bounds = tuple(zip(list(opt_lb.values()), upper_bounds))
        
    optimised = scipy.optimize.minimize(_composite_neg_ll, x0=np.array(list(opt_iv.values())),
                                                method=optimisation_algo,
                                                args=(X, parameter_names, verbose),
                                                bounds=bounds)
    inferred_params = _opt_to_model_params(dict(zip(parameter_names, optimised.x)))
    negll = optimised.fun

    assert optimised.success, f"Optimisation failed: {optimised.message}"

    return inferred_params, negll


def _coal_time_pdf(t, tau1, tau0, a, b, c1, c2, m1_prime, m2_prime, m1, m2, init_state):
    """ Only used for debugging"""

    i = init_state-1

    q1 = matrices.GeneratorMatrix(m1_prime=m1_prime, m2_prime=m2_prime, c1=c1, c2=c2, matrix_type="Q1")
    q2 = matrices.GeneratorMatrix(m1=m1, m2=m2, b=b, matrix_type="Q2")
    q3 = matrices.GeneratorMatrix(a=a, matrix_type="Q3")
    g, alpha = q1.left_eigenvectors()
    c, beta = q2.left_eigenvectors()
    ginv = matrices.GeneratorMatrix.inverse(g)
    cinv = matrices.GeneratorMatrix.inverse(g)
    p1 = matrices.TransitionMatrix(q1, q2, q3, t=tau1, tau1=tau1, tau0=tau0)
    p2 = matrices.TransitionMatrix(q1, q2, q3, t=(tau0-tau1), tau1=tau1, tau0=tau0)

    if t <= tau1:
        return float(-np.sum([alpha[k] * ginv[i,k] * g[k,3] * exp(-alpha[k]*t) for k in range(0,4)]))
    elif tau1 < t <= tau0:
        return float(-np.sum([[p1[i,j] * beta[k] * cinv[j,k] * c[k,3] * exp(-beta[k]*(t-tau1)) for k in range(0,4)] for j in range(0,3)]))
    elif t > tau0:
        return float(np.sum([[p1[i,j] * p2[j,l] * (1/a)*exp((-1/a)*(t-tau0)) for l in range(0,3)] for j in range(0,3)]))
    else:
        return 0