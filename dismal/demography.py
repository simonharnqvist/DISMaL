import numpy as np
import scipy.optimize
#import generate_demes
import utils
import likelihood
import metrics
import preprocess

class InferredDemography:

    def __init__(self, model_description,
                 a, b, c1, c2, tau1, tau0,  m1, m2, m1_prime, m2_prime, theta, negll, n_params):
        self.model_description = model_description
        self.a = a
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.tau0 = tau0
        self.tau1 = tau1
        self.m1 = m1
        self.m2 = m2
        self.m1_prime = m1_prime
        self.m2_prime = m2_prime
        self.theta = theta
        self.negll = negll
        self.n_params = n_params
        self.aic = metrics.aic(-negll, n_params)
        # self.demes_graph = generate_demes.create_demes_graph([a, b, c1, c2, tau0, tau1, m1, m2, m1_prime, m2_prime, theta],
        #                                                      model_description=self.model_description,
        #                                                      popnames=self.popnames)
        self.inferred_values = {'a':a, 'b':b, 'c1':c1, 'c2':c2, 'tau1':tau1,'tau0':tau0,
                                'm1':m1, 'm2':m2, 'm1_prime':m1_prime, 'm2_prime':m2_prime, 'theta':theta,
                                '-lnL':self.negll, 'aic':self.aic}

    def __repr__(self):
        return str(self.inferred_values)


class Demography:


    def __init__(self, X, model_description=None, model=None,
                 set_m1_zero = False, set_m2_zero = False, set_m1_prime_zero = False, set_m2_prime_zero = False, no_migration=False):

        
        if model.lower() in ["iso", "isolation"]:
            no_migration = True
            self.model = "iso"
        elif model.lower() in ["sc", "sec", "secondary_contact"]:
            set_m1_zero = True
            set_m2_zero = True
            self.model = "sc"
        elif model.lower() in ["iim", "initial_migration", "am", "ancestral_migration"]:
            set_m1_prime_zero = True
            set_m2_prime_zero = True
            self.model = "iim"
        elif model.lower() == "gim":
            self.model = "gim"
        else:
            if model is not None:
                raise ValueError(f"The model {model} is not available: please select from 'iso', 'sc', 'iim' or 'gim', or alternatively specify a bespoke model using the set_m1_zero, set_m2_zero, set_m1_prime_zero, and set_m2_prime_zero parameters")

        if no_migration:
            self.set_m1_zero = True
            self.set_m2_zero = True
            self.set_m1_prime_zero = True
            self.set_m2_prime_zero = True
        else:
            self.set_m1_zero = set_m1_zero
            self.set_m2_zero = set_m2_zero
            self.set_m1_prime_zero = set_m1_prime_zero
            self.set_m2_prime_zero = set_m2_prime_zero

        self.model_description = model_description
        self.X = X

    #     self.res = self.infer_parameters()
    #
    # def __repr__(self):
    #     return self.res


    def infer_parameters(self, optimisation_algo='L-BFGS-B', a_initial=1, b_initial=1, c1_initial=1, c2_initial=1,
                         tau0_initial=1, tau1_initial=2, m1_initial=0, m2_initial=0, m1_prime_initial=0, m2_prime_initial=0, theta_initial=5, 
                         a_lower=0.001, b_lower=0.001, c1_lower=0.001, c2_lower=0.001, tau0_lower=0.001, tau1_lower=0.001, m1_lower=0, m2_lower=0, 
                         m1_prime_lower=0, m2_prime_lower=0, theta_lower=0.0000001, verbose=True):
        

        initial_values = {"a":a_initial, "b":b_initial, "c1":c1_initial, "c2":c2_initial, "tau0":tau0_initial, "tau1":tau1_initial,
                        "m1":m1_initial, "m2":m2_initial, "m1_prime":m1_prime_initial, "m2_prime":m2_prime_initial, "theta":theta_initial}
        lower_bounds = {"a":a_lower, "b":b_lower, "c1":c1_lower, "c2":c2_lower, "tau0":tau0_lower, "tau1":tau1_lower,
                        "m1":m1_lower, "m2":m2_lower, "m1_prime":m1_prime_lower, "m2_prime":m2_prime_lower, "theta":theta_lower}

        if self.set_m1_prime_zero:
            initial_values.pop("m1_prime")
            lower_bounds.pop("m1_prime")
        if self.set_m2_prime_zero:
            initial_values.pop("m2_prime")
            lower_bounds.pop("m2_prime")
        if self.set_m1_zero:
            initial_values.pop("m1")
            lower_bounds.pop("m1")
        if self.set_m2_zero:
            initial_values.pop("m2")
            lower_bounds.pop("m2")
        
        n_params = len(list(initial_values.keys()))
        assert n_params == len(list(lower_bounds.keys()))

        inferred_params, negll = likelihood._optimise_negll(X=self.X, initial_vals=initial_values, lower_bounds=lower_bounds, optimisation_algo=optimisation_algo, verbose=verbose)
        
        for p in ["m1", "m2", "m1_prime", "m2_prime"]:
            if p not in list(inferred_params.keys()):
                inferred_params[p] = 0

        return InferredDemography(model_description=self.model_description, a=inferred_params["a"],
                                  b=inferred_params["b"], c1=inferred_params["c1"], c2=inferred_params["c2"],
                                  tau1=inferred_params["tau1"], tau0=inferred_params["tau0"],
                                  m1=inferred_params["m1"], m2=inferred_params["m2"], m1_prime=inferred_params["m1_prime"],
                                  m2_prime=inferred_params["m2_prime"], theta=inferred_params["theta"], negll=negll, n_params=n_params)











