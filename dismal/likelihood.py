import math
from math import factorial
from mpmath import exp
import numpy as np
from utils import opt_to_model_params
import matrices

# def pdf_q1(t, alpha, g, ginv, init_state, state):
#     """
#     PDF of the transition time for Q1.
#     :param float t: time
#     :param np.array alpha: vector of absolute left eigenvalues of Q1 matrix
#     :param sympy.Matrix g: matrix G of left eigenvectors of Q1
#     :param sympy.Matrix ginv: matrix inverse of G
#     :param int init_state: initial state of Markov chain in {1,2,3}
#     :return: PDF of absorption time
#     """
#     i = init_state-1
#     j = state-1

#     if state in [1,2,3]:
#         prob = float(np.sum([ginv[i,k]*g[k,j]*exp(-alpha[k]*t) for k in range(0,4)]))
#     elif state == 4:
#         prob = float(-np.sum([alpha[k]*ginv[i,k]*g[k,3]*exp(-alpha[k]*t) for k in range(0,4)]))
#     else:
#         raise ValueError("Invalid value for state")

#     return prob


# def pdf_q2(t, beta, c, cinv, init_state, state):
#     """
#     PDF of the transition time for Q2.
#     :param float t: time
#     :param np.array beta: vector of absolute left eigenvalues of Q2 matrix
#     :param sympy.Matrix c: matrix C of left eigenvectors of Q2
#     :param sympy.Matrix cinv: matrix inverse of C
#     :param int init_state: initial state of Markov chain in {1,2,3}
#     :return: PDF of absorption time
#     """
#     j = init_state-1
#     l = state-1
#     if state in [1,2,3]:
#         prob = float(np.sum([cinv[j,k]*c[k,l]*exp(-beta[k]*t) for k in range(0,4)]))
#     elif state == 4:
#         prob = float(-np.sum([beta[k]*cinv[j,k]*c[k,3]*exp(-beta[k]*t) for k in range(0,4)]))
#     else:
#         raise ValueError("Invalid value for state")

#     return prob

# def pdf_q3(t, a, state=4):
#     """
#     PDF of the absorption time for Q3 (other transition times are not needed).
#     :param float t: time
#     :param float a: relative size of population
#     :return: PDF of absorption time
#     """

#     if state == 4:
#         prob = float(1/a * exp(-1/a*t))
#     else:
#         raise ValueError("pdf_q3 not implemented for other values of state than 4")

#     return prob

def gs(s, theta, lmbda, t=None):
    """Geometric distribution of number of mutations"""

    gs_u = (lmbda * (theta**s)) / ((lmbda+theta)**(s+1))
    if t is None:
        return gs_u
    else:
        if s == 0:
            return exp(-theta * t) * gs_u * 1 # last term becomes 1 if s = 0 (but Python thinks it's 0 due to list comp)
        else:
            return exp(-theta * t) * gs_u * np.sum([(((lmbda+theta)**l)*(t**l))/math.factorial(l) for l in range(0,s)])


def pdf_s(s, a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta, init_state):
    """
    PDF of s, given parameter set
    :param float a: relative size of ancestral population
    :param float b: relative size of population 2 between time tau1 and tau0
    :param float c1: relative size of population 1 from present to tau1
    :param float c2: relative size of population 2 from present to tau1
    :param float tau0: split time
    :param float tau1: time at which gene flow changes, starts, or ceases (or neither), depending on model
    :param float m1: migration rate from population 1 to 2 between tau1 and tau0
    :param float m2: migration rate from population 2 to 1 between tau1 and tau0
    :param float m1_prime: migration rate from population 1 to 2 between present and tau0
    :param float m2_prime: migration rate from population 2 to 1 between present and tau0
    :param float theta: scaled mutation rate (4*Ne*mu)
    :param int state: state of Markov chain {1,2,3,4}
    :param int s: number of observed nucleotide differences
    :return: negll
    """
    i = init_state-1
    s = int(s)

    q1 = matrices.GeneratorMatrix(m1_prime=m1_prime, m2_prime=m2_prime, c1=c1, c2=c2, matrix_type="Q1")
    q2 = matrices.GeneratorMatrix(m1=m1, m2=m2, b=b, matrix_type="Q2")
    q3 = matrices.GeneratorMatrix(a=a, matrix_type="Q3")
    g, alpha = q1.left_eigenvectors()
    c, beta = q2.left_eigenvectors()
    ginv = matrices.GeneratorMatrix.inverse(g)
    cinv = matrices.GeneratorMatrix.inverse(c)
    p1 = matrices.TransitionMatrix(q1, q2, q3, t=tau1, tau1=tau1, tau0=tau0)
    p2 = matrices.TransitionMatrix(q1, q2, q3, t=(tau0-tau1), tau1=tau1, tau0=tau0)

    # Lists of expectations
    w_expect = np.array([gs(s=s, theta=theta, lmbda=eigval) for eigval in alpha])
    tau1_w_expect = np.array([gs(s=s, theta=theta, lmbda=eigval, t=tau1) for eigval in alpha])
    tau1_y_expect = np.array([gs(s=s, theta=theta, lmbda=eigval, t=tau1) for eigval in beta])
    tau0_y_expect = np.array([gs(s=s, theta=theta, lmbda=eigval, t=tau0) for eigval in beta])
    tau0_x_expect = np.array(gs(s=s, theta=theta, lmbda=1/a, t=tau0))

    term1 = np.sum([float(ginv[i,k]) * float(g[k,3]) * (w_expect[k] - tau1_w_expect[k] *
                                          exp(-alpha[k]*tau1)) for k in range(0,4) if alpha[k] > 0])

    term2 = np.sum([[float(p1[i,j]) * float(cinv[j,k]) * float(c[k,3]) * (tau1_y_expect[k] - tau0_y_expect[k] *
                                            exp(-beta[k]*(tau0-tau1))) for k in range(0,4) if beta[k] > 0]
                   for j in range(0,3)])

    term3 = np.sum([[float(p1[i,j]) * float(p2[j,l]) * tau0_x_expect for l in range(0,3)] for j in range(0,3)])

    pr_s = float(-term1-term2+term3)

    return pr_s


def composite_neg_ll(params, X, verbose=True):
    """
    Calculate the composite negative log-likelihood of a parameter set, given a dataset X.
    :param list params: Parameters [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]
    :param np.array X: array [x1, x2, x3] where X[i] is a dictionary of s_counts per state
    :return: composite_neg_ll


    """
    # Check for NaNs in input
    params = [0 if math.isnan(param) else param for param in params]

    # Convert params for likelihood function
    model_params = opt_to_model_params(params)
    assert not any([np.isnan(param) for param in model_params]), f"NaN values in converted model params {model_params}; original params {params}"

    a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta = model_params

    # Multiply each s by likelihood of that s, and sum to generate composite LL
    composite_neg_ll = -np.sum([np.sum([int(X[i-1][s]) * np.log(pdf_s(a=a, b=b, c1=c1, c2=c2, tau0=tau0, tau1 = tau1,
                                                                   m1=m1, m2=m2, m1_prime=m1_prime, m2_prime=m2_prime,
                                                                   theta=theta, init_state=i, s=s)) for s in X[i-1].keys()]) for i in [1,2,3]])

    if verbose:
        print(f"Parameters: {model_params}, -lnL: {composite_neg_ll}")

    return composite_neg_ll


def coal_time_pdf(t, tau1, tau0, a, b, c1, c2, m1_prime, m2_prime, m1, m2, init_state):
    """
    PDF of coalescence time.
    :param t: time t
    :param float tau0: split time
    :param float tau1: time at which gene flow changes, starts, or ceases (or neither), depending on model
    :param float a: relative size of ancestral population
    :param float b: relative size of population 2 between time tau1 and tau0
    :param float c1: relative size of population 1 from present to tau1
    :param float c2: relative size of population 2 from present to tau1
    :param float m1_prime: migration rate from population 1 to 2 between present and tau0
    :param float m2_prime: migration rate from population 2 to 1 between present and tau0
    :param float m1: migration rate from population 1 to 2 between tau1 and tau0
    :param float m2: migration rate from population 2 to 1 between tau1 and tau0
    :param int init_state: initial state of Markov chain {1,2,3,4}
    :return: coal_time_pdf
    """
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