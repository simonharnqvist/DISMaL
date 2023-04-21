import numpy as np



def _model_to_opt_params(model_params):
    """
    Reparameterise from model parameters to optimisation parameters.
    :param list params: parameters [a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta]
    :return: [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]
    """

    params = list(model_params.keys())
    opt_params = {}
    
    if "theta" in params:
        opt_params["theta1"] = model_params["theta"]
    if "a" in params:
        opt_params["theta0"] = model_params["a"]*model_params["theta"]
    if "b" in params:
        opt_params["theta2"] = model_params["b"]*model_params["theta"]
    if "c1" in params:
        opt_params["theta1_prime"] = model_params["c1"]*model_params["theta"]
    if "c2" in params:
        opt_params["theta2_prime"] = model_params["c2"]*model_params["theta"]
    if "tau1" in params:
        opt_params["t1"] = model_params["tau1"]*model_params["theta"]
    if "tau0" in params:
        opt_params["v"] = model_params["theta"]*(model_params["tau0"]-model_params["tau1"])
    if "m1" in params:
        opt_params["m1_star"] = model_params["m1"]
    if "m2" in params:
        opt_params["m2_star"] = model_params["m2"] * model_params["b"]
    if "m1_prime" in params:
        opt_params["m1_prime_star"] = model_params["c1"] * model_params["m1_prime"]
    if "m2_prime" in params:
        opt_params["m2_prime_star"] = model_params["c2"] * model_params["m2_prime"]

    return opt_params

def _opt_to_model_params(opt_params):
    """
    Reparameterise back from optimisation parameters to model parameters.
    :param list params: [theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star]
    :return: [a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta]
    """

    params = list(opt_params.keys())
    model_params = {}


    if "theta1" in params:
        if opt_params["theta1"] == 0:
            opt_params["theta1"] = 1e-5
        model_params["theta"] = opt_params["theta1"]

    if "theta0" in params:
        model_params["a"] = opt_params["theta0"] / opt_params["theta1"]
    if "theta2" in params:
        model_params["b"] = opt_params["theta2"] / opt_params["theta1"]
        if model_params["b"] == 0:
            model_params["b"] = 1e-5
    if "theta1_prime" in params:
        model_params["c1"] = opt_params["theta1_prime"] / opt_params["theta1"]
        if model_params["c1"] == 0:
            model_params["c1"] = 1e-5
    if "theta2_prime" in params:
        model_params["c2"] = opt_params["theta2_prime"] / opt_params["theta1"]
        if model_params["c2"] == 0:
            model_params["c2"] = 1e-5
    if "t1" in params:
        model_params["tau1"] = opt_params["t1"] / opt_params["theta1"]
    if "v" in params:
        model_params["tau0"] = (opt_params["v"] + opt_params["t1"])/opt_params["theta1"]
    if "m1_star" in params:
        model_params["m1"] = opt_params["m1_star"]
    if "m2_star" in params:
        model_params["m2"] = opt_params["m2_star"] / (opt_params["theta2"] / opt_params["theta1"])
    if "m1_prime_star" in params:
        model_params["m1_prime"] = opt_params["theta1"] * opt_params["m1_prime_star"] / opt_params["theta1_prime"]
    if "m2_prime_star" in params:
        model_params["m2_prime"] = opt_params["theta1"] * opt_params["m2_prime_star"] / opt_params["theta2_prime"]


    return model_params