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


def make_generator_matrices(theta0, theta1, theta2, theta1_prime, theta2_prime,
                             m1_star=None, m2_star=None, m1_prime_star=None, m2_prime_star=None):
    """Make generator matrices Q1, Q2, Q3"""
    
    m1_star = 0 if m1_star is None else m1_star
    m2_star = 0 if m2_star is None else m2_star
    m1_prime_star = 0 if m1_prime_star is None else m1_prime_star
    m2_prime_star = 0 if m2_prime_star is None else m2_prime_star
    assert not None in [theta0, theta1, theta2, theta1_prime, theta2_prime, m1_star, m2_star, m1_prime_star, m2_prime_star]
    assert theta0 > 0, "Theta0 must be greater than 0"

    # Convert params
    a, b, c1, c2 = [popsize/theta1 for popsize in [theta0, theta2, theta1_prime, theta2_prime]]
    assert all([i > 0 for i in [a, b, c1, c2]]), "Converted population sizes must be greater than 0"
    m1 = m1_star
    m2 = m2_star/b
    m1_prime = m1_prime_star/c1
    m2_prime = m2_prime_star/c2

    q1 = matrices.GeneratorMatrix(m1_prime=m1_prime, m2_prime=m2_prime, c1=c1, c2=c2, matrix_type="Q1")
    q2 = matrices.GeneratorMatrix(m1=m1, m2=m2, b=b, matrix_type="Q2")
    q3 = matrices.GeneratorMatrix(a=a, matrix_type="Q3")

    return q1, q2, q3

def make_transition_matrices(q1, q2, q3, t1, v, theta1):
    """Make transition matrices P1, P2"""

    tau1 = t1/theta1
    tau0 = (t1+v)/theta1
    assert tau0 >= tau1, "tau1 greater than tau0"

    p1 = matrices.TransitionMatrix(q1, q2, q3, t=tau1, tau1=tau1, tau0=tau0)
    p2 = matrices.TransitionMatrix(q1, q2, q3, t=(tau0-tau1), tau1=tau1, tau0=tau0)

    return p1, p2



def _pdf_s(s, state, theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star):
    """ Probability density function of s, the number of nucleotide differences between two blocks.

    Keyword arguments:
    s -- number of nucleotide differences
    [params go here]
    theta -- effective population size 4*Ne*mu
    state -- sampling state {1,2,3} where 1 means that both blocks are sampled from population 1, 2 means that both blocks are sampled from population 2, and 3 means one block per population
    """

    i = state-1
    s = int(s)

    q1, q2, q3 = make_generator_matrices(theta0=theta0, theta1=theta1, theta2=theta2, theta1_prime=theta1_prime,
                                         theta2_prime=theta2_prime, m1_star=m1_star, m2_star=m2_star,
                                         m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
    g, alpha = q1.left_eigenvectors()
    c, beta = q2.left_eigenvectors()
    ginv = matrices.GeneratorMatrix.inverse(g)
    cinv = matrices.GeneratorMatrix.inverse(c)
    #p1, p2 = make_transition_matrices(q1, q2, q3, t1, v, theta1)

    # Lists of expectations; eigenvalues are rate parameters of nt difference distribution
    tau1 = t1/theta1
    tau0 = (t1+v)/theta1
    a = theta0/theta1

    w_expect = np.array([_pdf_gs(s=s, theta=theta1, lmbda=eigval) for eigval in alpha])
    tau1_w_expect = np.array([_pdf_gs(s=s, theta=theta1, lmbda=eigval, t=tau1) for eigval in alpha])
    tau1_y_expect = np.array([_pdf_gs(s=s, theta=theta1, lmbda=eigval, t=tau1) for eigval in beta])
    tau0_y_expect = np.array([_pdf_gs(s=s, theta=theta1, lmbda=eigval, t=tau0) for eigval in beta])
    tau0_x_expect = np.array(_pdf_gs(s=s, theta=theta1, lmbda=1/a, t=tau0))

    term1 = np.sum([float(ginv[i,k]) * float(g[k,3]) * (w_expect[k] - tau1_w_expect[k] *
                                          exp(-alpha[k]*tau1)) for k in range(0,4) if alpha[k] > 0])

    term2 = np.sum([matrices.p1(i=i, j=j, g=g, ginv=ginv, alpha=alpha, t=tau1) * 
                    np.sum([float(cinv[j,k]) * float(c[k,3]) * (tau1_y_expect[k] - tau0_y_expect[k] *
                                            exp(-beta[k]*(tau0-tau1))) for k in range(0,4) if beta[k] > 0]) for j in range(0,3)])

    term3 = np.sum([[matrices.p1(i=i, j=j, g=g, ginv=ginv, alpha=alpha, t=tau1) *
                     matrices.p2(j=j, l=l, c=c, cinv=cinv, beta=beta, t=(tau0-tau1)) *
                       tau0_x_expect for j in range(0,3)] for l in range(0,3)])

    pr_s = float(-term1-term2+term3)

    assert 0 <= pr_s <= 1, f"Probabilities outside range [0,1]: p={pr_s}, term1={term1}, term2={term2}, term3={term3} for parameters {[s, state, theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]}"

    return pr_s

def p1(i,j,k, g, ginv, alpha, t):
    return ginv[i,k] * g[k,j] * math.exp(-alpha[k]*t)

<<<<<<< HEAD
def _sval_likelihood(s_val, s_count, params, state):
   
=======
def p2(j,l, k, c, cinv, beta, t):
    return cinv[j,k] * c[k,l] * math.exp(-beta[k]*t)

# def _pdf_s_new(s, state, theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star):

#     i = state-1
#     s = int(s)

#     q1, q2, q3 = make_generator_matrices(theta0=theta0, theta1=theta1, theta2=theta2, theta1_prime=theta1_prime,
#                                          theta2_prime=theta2_prime, m1_star=m1_star, m2_star=m2_star,
#                                          m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
#     g, alpha = q1.left_eigenvectors()
#     c, beta = q2.left_eigenvectors()
#     ginv = matrices.GeneratorMatrix.inverse(g)
#     cinv = matrices.GeneratorMatrix.inverse(c)

#     a = theta0/theta1
#     theta = theta1
#     tau1 = t1/theta1
#     tau0 = (t1+v)/theta1

#     terms1 = np.sum([[ginv[i,k] * g[k,3] * (alpha[k]*(theta**s))/((alpha[k]+theta)**(s+1))] for k in range(0,4)])
    
#     terms2 = np.sum([(1-exp(-(alpha[k]+theta)*tau1) * np.sum([((alpha[k]+theta)**l) * (tau1**l)/factorial(l) for l in range(0,s+1)])) for k in range(0, len(alpha)) if alpha[k] > 0])
    
#     terms3 = np.sum([[np.sum([p1(i=i, j=j, k=k, g=g, ginv=ginv, alpha=alpha, t=tau1) for k in range(0,3)]) *
#                         cinv[j,k] * c[k,3] * (beta[k] * (theta**s))/((beta[k]+theta)**(s+1)) for j in range(0,3)] for k in range(0,4) if beta[k] > 0])
    
#     terms4 = np.sum([exp(-theta*tau1) * np.sum([((beta[k]+theta)**l) * ((tau1**l)/factorial(l)) for l in range(0,s+1)]) for k in range(0,4)])

#     terms5 = np.sum([[[p1(i=i, j=j, k=k, g=g, ginv=ginv, alpha=alpha, t=tau1) * p2(j=j, l=l, k=k, c=c, cinv=cinv, beta=beta, t=(tau0-tau1)) for l in range(0,3)] for j in range(0,3)] for k in range(0,4)])

#     terms6 = exp(-theta * tau0) * ((a*theta)**s) / ((1+a*theta)**(s+1))

#     terms7 = np.sum([(((1/a)+theta)**l) * (tau0**l) / factorial(l) for l in range(0, s+1)])



#     pr_s = float(-terms1 - (terms2 * (terms3-terms4)) + terms5 * terms6 * terms7)
#     return pr_s
    

                                                                                    


def _sval_likelihood(s_count, s, state, theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star):
>>>>>>> 26db1e7 (new ll fxn implementation)
    """ Calculate the likelihood of a given value of s.
    KL: careful with the wording: a value of S has a probability (given parameters)
    parameters have a likelihood (given a values of S).... 

    Keyword arguments:
    s_val -- the number of nucleotide differences (the likelihood of which is calculated)
    s_counts -- the count of that value of s (i.e. how many times does that value of s occur in that state)
    params -- dictionary of model parameters
    state -- sampling state {1,2,3} where 1 means that both blocks are sampled from population 1, 2 means that both blocks are sampled from population 2, and 3 means one block per population
    """

    p_s = _pdf_s(s=s, state=state, theta0=theta0, theta1=theta1, theta2=theta2, theta1_prime=theta1_prime, theta2_prime=theta2_prime,
                  t1=t1, v=v, m1_star=m1_star, m2_star=m2_star, m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
    assert 0 <= p_s <= 1, f"Probabilities outside range [0,1]: p={p_s} for parameters {s, state, theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star}"

    logL = np.log(p_s) * s_count
    assert logL < 0, "Positive log-likelihood detected - (positive) log-likelihoods must be negative"
    return logL


<<<<<<< HEAD
def _composite_neg_ll(params, X, parameter_names, verbose=True):
    """ Calculate the composite log likelihood of a parameter set, given dataset X.
=======
def _composite_neg_ll(param_vals, param_names, X, verbose=True):
    """ Calculate the composite negative log likelihood of a parameter set, given dataset X.
>>>>>>> 26db1e7 (new ll fxn implementation)

    Keyword arguments:
    param_vals -- parameters [theta0, theta1, theta2, theta1_prime, theta2_prime, t1] and optionally [m1_star, m2_star, m1_prime_star, m2_prime_star]
    param_names -- list of names of parameters
    X -- list of three dictionaries, each containing key-value pairs where the key is s, the number of differences, and the value is the corresponding count
    verbose -- whether to output parameter values and log-likelihoods during fitting
    """
    # Check for NaNs in input
    param_vals = [0 if math.isnan(param) else float(param) for param in param_vals]
    param_dict = dict(zip(param_names, param_vals))
    assert isinstance(param_dict, dict)

    # Set missing mig rates to 0
    for param in ["m1_star", "m2_star", "m1_prime_star", "m2_prime_star"]:
        if param not in list(param_dict.keys()):
            param_dict[param] = 0

    theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star = list(param_dict.values())
    print(param_dict)

<<<<<<< HEAD
    assert isinstance(model_params, dict)
    # Multiply each s count by the probability of that s, and sum to generate composite LL
    log_likelihoods = list(itertools.chain(*[[_sval_likelihood(s_val = s, s_count=X[state-1][s], params = model_params, state=state) for s in X[state-1].keys()] 
=======
    # Multiply each s by likelihood of that s, and sum to generate composite LL (flatten list with itertools)
    log_likelihoods = list(itertools.chain(*[[_sval_likelihood(s_count=X[state-1][s], s=s, state=state,
                                                                theta0=theta0, theta1=theta1, theta2=theta2,
                                                                  theta1_prime=theta1_prime, theta2_prime=theta2_prime,
                                                                    t1=t1, v=v, m1_star=m1_star, m2_star=m2_star,
                                                                      m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
                                              for s in X[state-1].keys()] 
>>>>>>> 26db1e7 (new ll fxn implementation)
                                         for state in [1,2,3]]))
    
    assert all(i < 0 for i in log_likelihoods), f"Positive log-likelihood detected - (positive) log-likelihoods must be negative: {log_likelihoods}"
    negll = -np.sum(log_likelihoods)
    assert negll > 0, "Negative -negll detected; negative log-likelihoods must be positive"

    if verbose:
        print(f"Parameters: {param_dict}, -lnL: {negll}")

    return negll

def _optimise_negll(X, initial_vals, lower_bounds, optimisation_algo, verbose):
    """ Optimise negative log likelihood.
        
    Keyword arguments:
    X -- list of dictionaries of s values
    initial_vals -- dict of initial values
    lower_bounds -- dict of lower bounds
    optimisation_algo -- which algo to use (see scipy.optimize.minimize)
    """
    param_names = list(initial_vals.keys())
    upper_bounds = [None] * len(param_names)
    bounds = tuple(zip(list(lower_bounds.values()), upper_bounds))

    print(initial_vals)
    print(lower_bounds)
        
    optimised = scipy.optimize.minimize(_composite_neg_ll, x0=np.array(list(initial_vals.values())),
                                                method=optimisation_algo,
                                                args=(param_names, X, verbose),
                                                bounds=bounds)
    inferred_params = optimised.x
    negll = optimised.fun

    assert optimised.success, f"Optimisation failed: {optimised.message}"

    return inferred_params, negll



# def _coal_time_pdf(t, tau1, tau0, a, b, c1, c2, m1_prime, m2_prime, m1, m2, init_state):
#     """ Only used for debugging"""

#     i = init_state-1

<<<<<<< HEAD
    if t <= tau1:
        return float(-np.sum([alpha[k] * ginv[i,k] * g[k,3] * exp(-alpha[k]*t) for k in range(0,4)]))
    elif tau1 < t <= tau0:
        return float(-np.sum([[p1[i,j] * beta[k] * cinv[j,k] * c[k,3] * exp(-beta[k]*(t-tau1)) for k in range(0,4)] for j in range(0,3)]))
    elif t > tau0:
        return float(np.sum([[p1[i,j] * p2[j,l] * (1/a)*exp((-1/a)*(t-tau0)) for l in range(0,3)] for j in range(0,3)]))
    else:
        return 0
=======
#     q1 = matrices.GeneratorMatrix(m1_prime=m1_prime, m2_prime=m2_prime, c1=c1, c2=c2, matrix_type="Q1")
#     q2 = matrices.GeneratorMatrix(m1=m1, m2=m2, b=b, matrix_type="Q2")
#     q3 = matrices.GeneratorMatrix(a=a, matrix_type="Q3")
#     g, alpha = q1.left_eigenvectors()
#     c, beta = q2.left_eigenvectors()
#     ginv = matrices.GeneratorMatrix.inverse(g)
#     cinv = matrices.GeneratorMatrix.inverse(g)
#     p1 = matrices.TransitionMatrix(q1, q2, q3, t=tau1, tau1=tau1, tau0=tau0)
#     p2 = matrices.TransitionMatrix(q1, q2, q3, t=(tau0-tau1), tau1=tau1, tau0=tau0)

#     if t <= tau1:
#         return float(-np.sum([alpha[k] * ginv[i,k] * g[k,3] * exp(-alpha[k]*t) for k in range(0,4)]))
#     elif tau1 < t <= tau0:
#         return float(-np.sum([[p1[i,j] * beta[k] * cinv[j,k] * c[k,3] * exp(-beta[k]*(t-tau1)) for k in range(0,4)] for j in range(0,3)]))
#     elif t > tau0:
#         return float(np.sum([[p1[i,j] * p2[j,l] * (1/a)*exp((-1/a)*(t-tau0)) for l in range(0,3)] for j in range(0,3)]))
#     else:
#         return 0
>>>>>>> 26db1e7 (new ll fxn implementation)
