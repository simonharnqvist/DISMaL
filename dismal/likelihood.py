import numpy as np
import scipy
from scipy import linalg
from scipy.stats import poisson
from generator_matrices import GeneratorMatrix

np.set_printoptions(suppress=True)

def stochastic_matrix(eigenvect_mat, inv_eigenvect_mat, eigenvals, t):
    """ Fast matrix exponentiation to produce stochastic matrix from eigenvalues and eigenvectors of transition rate matrix. Equivalent to linalg.expm(Q), but faster if eigenvalues+vectors are already available.

    Args:
        matrix (_type_): _description_
        inv_matrix (_type_): _description_
        eigenvalues (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    return inv_eigenvect_mat @ np.diag(np.exp(eigenvals * t)) @ eigenvect_mat

def prob_s_events_matrix(alpha, beta, gamma, s_vals, t1, t0, rel_mu=1):
    """Calculate probability of seeing s events during a given time span; returns matrix s x i, where each eigenvalue corresponds to a state (?)"""
    pass

def alpha_matrix(alpha, s_vals, t1, rel_mu=1):
    """Generate matrix of s values (columns) x adjusted alpha values (rows). This corresponds to P(s events before coalescence) * 1-P(s events)"""
    alpha, s_vals = np.array(alpha), np.array(s_vals)
    alphas = []
    for s in s_vals:
        alphas.append((alpha/(alpha+rel_mu)) * ((rel_mu/(alpha+rel_mu))**s) * (1-poisson.cdf(s, (t1*(alpha+rel_mu)))))
    return np.transpose(np.array(alphas))


def beta_matrix(beta, s_vals, t1, t0, rel_mu=1):
    beta, s_vals = np.array(beta), np.array(s_vals)
    betas = []
    for s in s_vals:
        betas.append((beta/(beta+rel_mu)) * ((rel_mu/(beta+rel_mu))**s) *
                      np.exp(beta*t1) * (poisson.cdf(s, (t1*(beta+rel_mu))) - poisson.cdf(s, (t0*(beta+rel_mu)))))
    return np.transpose(np.array(betas))

def gamma_matrix(gamma, s_vals, t0, rel_mu=1):
    gamma, s_vals = np.array(gamma), np.array(s_vals)
    gammas = []
    for s in s_vals:
        gammas.append((gamma/(gamma+rel_mu)) * ((rel_mu/(gamma+rel_mu))**s) *
                       np.exp(gamma*t0) * poisson.cdf(s, (t0*(gamma+rel_mu))))
    return np.transpose(np.array(gammas))

def likelihood_matrix(q1, q2, q3, t1, v, S=None, s_vals=None):
    """S is a matrix of counts; return matrix of likelihood of params given s (columns) and state (rows)"""
    g = q1.eigenvectors
    c = q2.eigenvectors
    ginv = q1.eigenvectors_inv
    cinv = q2.eigenvectors_inv
    alpha = -q1.eigenvalues[0:3]
    beta = -q2.eigenvalues[0:3]
    gamma = -q3[0,0]

    gg = -ginv @ np.diag(g[:,3])
    cc = -cinv @ np.diag(c[:,3])
    pij1 = stochastic_matrix(eigenvect_mat=g, inv_eigenvect_mat=ginv, eigenvals=q1.eigenvalues, t=t1)
    pij2 = stochastic_matrix(eigenvect_mat=c, inv_eigenvect_mat=cinv, eigenvals=q2.eigenvalues, t=v)

    if s_vals is None:
        assert S is not None
        s_vals = [s for s in range(0, S.shape[1])]

    t0 = t1+v

    ll_matrix = [-np.log(gg[i, 0:3] @ alpha_matrix(alpha=alpha, s_vals=s_vals, t1=t1)
                         + pij1[i, 0:3] @ cc[0:3, 0:3] @ beta_matrix(beta=beta, s_vals=s_vals, t1=t1, t0=t0) +
                         (1 - (pij1@pij2)[i,3]) * gamma_matrix(gamma=gamma, s_vals=s_vals, t0=t0)) for i in [0, 1, 2]]
    
    return ll_matrix

def composite_neg_ll(params, S, verbose=False):
    """S = matrix of s_counts per state and s_value"""

    theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star = params

    q1 = GeneratorMatrix(matrix_type='Q1', theta1=theta1, theta1_prime=theta1_prime, theta2_prime=theta2_prime,
                          m1_prime_star=m1_prime_star, m2_prime_star=m2_prime_star)
    q2 = GeneratorMatrix(matrix_type='Q2', theta1=theta1, theta2=theta2, m1_star=m1_star, m2_star=m2_star)
    q3 = GeneratorMatrix(matrix_type='Q3', theta1=theta1, theta0=theta0)

    ll_matrix = likelihood_matrix(q1=q1, q2=q2, q3=q3, t1=t1, v=v, S=S)
    
    negll = np.sum(ll_matrix * S)

    if verbose:
        print(f"Parameters: {params}, -lnL: {negll}")
    
    return negll


def optimise_neg_ll(S, initial_vals, lower_bounds, upper_bounds, optimisation_algo, verbose):
    print(f"iv: {initial_vals}, lb: {lower_bounds}, ub: {upper_bounds}")

    bounds = tuple(zip(lower_bounds, upper_bounds))

    optimised = scipy.optimize.minimize(composite_neg_ll, x0=np.array(initial_vals),
                                                method=optimisation_algo,
                                                args=(S, verbose),
                                                bounds=bounds)
    inferred_params = optimised.x
    negll = optimised.fun

    assert optimised.success, f"Optimisation failed: {optimised.message}"

    return inferred_params, negll
