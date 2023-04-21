import numpy as np
from scipy import linalg
from scipy.stats import poisson

def matrix_manipulations(matrix):
    eigenvals, eigenvects = linalg.eig(matrix, left=False, right=True)
    eigenvects = eigenvects.transpose()
    inv = linalg.inv(matrix)
    return eigenvals, eigenvects, inv

def p_matrix(matrix, inv_matrix, eigenvalues, t):
    return inv_matrix @ np.diag(np.exp(eigenvalues * t)) @ matrix

def alpha_matrix(alpha, s_vals, t1, rel_mu=1):
    alpha, s_vals = np.array(alpha), np.array(s_vals)
    alphas = []
    for s in s_vals:
        alphas.append((alpha/(alpha+rel_mu)) * ((rel_mu/(alpha+rel_mu))**s) * (1-poisson.cdf(s, (t1*(alpha+rel_mu)))))
    return np.transpose(np.array(alphas))

def beta_matrix(beta, s_vals, t1, t0, rel_mu=1):
    beta, s_vals = np.array(beta), np.array(s_vals)
    betas = []
    for s in s_vals:
        betas.append((beta/(beta+rel_mu)) * ((rel_mu/(beta+rel_mu))**s) * np.exp(beta*t1) * (poisson.cdf(s, (t1*(beta+rel_mu))) - poisson.cdf(s, (t0*(beta+rel_mu)))))
    return np.transpose(np.array(betas))

def gamma_matrix(gamma, s_vals,  t0, rel_mu=1):
    gamma, s_vals = np.array(gamma), np.array(s_vals)
    gammas = []
    for s in s_vals:
        gammas.append((gamma/(gamma+rel_mu)) * ((rel_mu/(gamma+rel_mu))**s) * np.exp(gamma*t0) * poisson.cdf(s, (t0*(gamma+rel_mu))))
    return np.transpose(np.array(gammas))

def likelihood_matrix(q1, q2, q3, t1, v, S):
    """Produce matrix [s,i] = L, where L is the likelihood of parameters given s nt differences and state i """

    g, eigvals_q1 = q1.eigen()
    c, eigvals_q2 = q2.eigen()
    ginv = linalg.inv(g)
    cinv = linalg.inv(c)
    alpha = -eigvals_q1[0:3]
    beta = -eigvals_q2[0:3]
    gamma = 1 #1/q3.pop_size1 #1/a

    gg = ginv @ np.diag(g[:,3])
    cc = cinv @ np.diag(c[:,3])
    pij1 = p_matrix(g, ginv, eigvals_q1, t1)
    pij2 = p_matrix(c, cinv, eigvals_q2, v)

    ll_matrix = [] # shape = s_val x state
    for i in [0, 1, 2]: # per state
        ll_matrix.append(gg[i, 0:3] @ alpha_matrix(alpha, S[i], t1)
                                + pij1[i, 0:3] @ cc[0:3, 0:3] @ beta_matrix(beta, S[i], t1=t1, t0=(v+t1)) +
                                (1 - (pij1@pij2)[i,3]) * gamma_matrix(gamma,S[i],t1+v))
    
    return ll_matrix


