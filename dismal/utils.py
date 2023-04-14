import numpy as np



def model_to_opt_params(params):
    """
    Reparameterise from model parameters to optimisation parameters.
    :param list params: parameters [a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta]
    :return: [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]
    """
    a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta = params
    theta0 = a*theta
    theta1 = theta
    theta2 = b*theta
    theta1_prime = c1*theta
    theta2_prime = c2*theta
    t1 = theta * tau1
    v = theta*(tau0-tau1)
    m1_star = m1
    m2_star = b*m2
    m1_prime_star = c1*m1_prime
    m2_prime_star = c2*m2_prime

    return [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]

def opt_to_model_params(params):
    """
    Reparameterise back from optimisation parameters to model parameters.
    :param list params: [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]
    :return: [a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta]
    """
    theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star = params

    if theta1 == 0:
        theta1 = 1e-5

    theta = theta1
    a = theta0/theta
    b = theta2/theta
    if b == 0:
        b = 1e-5
    c1 = theta1_prime/theta
    c2 = theta2_prime/theta
    if c1 == 0:
        c1 = 1e-5
    if c2 == 0:
        c2 = 1e-5
    tau1 = t1/theta
    tau0 = v/theta+tau1
    m1 = m1_star
    m2 = m2_star/b
    m1_prime = m1_prime_star/c1
    m2_prime = m2_prime_star/c2

    model_params = [a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta]

    assert not any([np.isnan(param) for param in model_params]), f"NaNs found when converting from optimisation to model params; opt_params:{params}, model_params:{model_params}"

    return model_params