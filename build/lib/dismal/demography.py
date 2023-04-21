import numpy as np
import scipy.optimize
#import generate_demes
import utils
import likelihood
import metrics
import preprocess

class InferredDemography:

    def __init__(self, theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v,
                  m1_star, m2_star, m1_prime_star, m2_prime_star, negll, n_params):
        self.negll = negll
        self.n_params = n_params
        self.aic = metrics.aic(-negll, n_params)
        # self.demes_graph = generate_demes.create_demes_graph([a, b, c1, c2, tau0, tau1, m1, m2, m1_prime, m2_prime, theta],
        #                                                      model_description=self.model_description,
        #                                                      popnames=self.popnames)
        self.inferred_values = {"theta0":theta0, "theta1":theta1, "theta2":theta2, "theta1_prime":theta1_prime, "theta2_prime":theta2_prime,
                                 "t1":t1, "v":v, "m1_star":m1_star, "m2_star":m2_star, "m1_prime_star":m1_prime_star, "m2_prime_star":m2_prime_star,
                                '-lnL':self.negll, 'aic':self.aic}

    def __repr__(self):
        return str(self.inferred_values)


class Demography:


    def __init__(self, X, model=None,
                 set_m1_star_zero = False, set_m2_star_zero = False,
                   set_m1_prime_star_zero = False, set_m2_prime_star_zero = False, no_migration=False):

        
        if model.lower() in ["iso", "isolation"]:
            no_migration = True
            self.model = "iso"
        elif model.lower() in ["sc", "sec", "secondary_contact"]:
            set_m1_star_zero = True
            set_m2_star_zero = True
            self.model = "sc"
        elif model.lower() in ["iim", "initial_migration", "am", "ancestral_migration"]:
            set_m1_prime_star_zero = True
            set_m2_prime_star_zero = True
            self.model = "iim"
        elif model.lower() == "gim":
            self.model = "gim"
        else:
            if model is not None:
                raise ValueError(f"The model {model} is not available: please select from 'iso', 'sc', 'iim' or 'gim', or alternatively specify a bespoke model")

        if no_migration:
            self.set_m1_star_zero = True
            self.set_m2_star_zero = True
            self.set_m1_prime_star_zero = True
            self.set_m2_prime_star_zero = True
        else:
            self.set_m1_star_zero = set_m1_star_zero
            self.set_m2_star_zero = set_m2_star_zero
            self.set_m1_prime_star_zero = set_m1_prime_star_zero
            self.set_m2_prime_star_zero = set_m2_prime_star_zero

        self.X = X

    #     self.res = self.infer_parameters()
    #
    # def __repr__(self):
    #     return self.res


    def infer_parameters(self, optimisation_algo='L-BFGS-B', theta0_iv=5, theta1_iv=5, theta2_iv=5, theta1_prime_iv=5, theta2_prime_iv=5,
                         t1_iv=5, v_iv=5, m1_star_iv=0.3, m2_star_iv=0.3, m1_prime_star_iv=0.3, m2_prime_star_iv=0.3,
                         theta0_lb=0.01, theta1_lb=0.01, theta2_lb=0.01, theta1_prime_lb=0.01, theta2_prime_lb=0.01,
                         t1_lb=0, v_lb=0, m1_star_lb=0, m2_star_lb=0, m1_prime_star_lb=0, m2_prime_star_lb=0, verbose=True):
        

        initial_values = {"theta0":theta0_iv, "theta1":theta1_iv, "theta2":theta2_iv, "theta1_prime":theta1_prime_iv, "theta2_prime":theta2_prime_iv,
                                 "t1":t1_iv, "v":v_iv, "m1_star":m1_star_iv, "m2_star":m2_star_iv, "m1_prime_star":m1_prime_star_iv, "m2_prime_star":m2_prime_star_iv}
        lower_bounds = {"theta0":theta0_lb, "theta1":theta1_lb, "theta2":theta2_lb, "theta1_prime":theta1_prime_lb, "theta2_prime":theta2_prime_lb,
                                 "t1":t1_lb, "v":v_lb, "m1_star":m1_star_lb, "m2_star":m2_star_lb, "m1_prime_star":m1_prime_star_lb, "m2_prime_star":m2_prime_star_lb}


        # Remove migration rate from model if disallowed
        constraints = [self.set_m1_star_zero, self.set_m2_star_zero, self.set_m1_prime_star_zero, self.set_m2_prime_star_zero]
        mig_params = ["m1_star", "m2_star", "m1_prime_star", "m2_prime_star"]
        for i in range(0,4):
                if constraints[i]:
                    initial_values.pop(mig_params[i])
                    lower_bounds.pop(mig_params[i])
        
        n_params = len(list(initial_values.keys()))
        assert n_params == len(list(lower_bounds.keys()))

        print(f"params in model: {list(initial_values.keys())}, with initial values {list(initial_values.values())} and lower bounds {list(lower_bounds.values())}")

        p, negll = likelihood._optimise_negll(X=self.X, initial_vals=initial_values, lower_bounds=lower_bounds, optimisation_algo=optimisation_algo, verbose=verbose)
        inferred_params = dict(zip(list(initial_values.keys()), p))
        
        for p in ["m1_star", "m2_star", "m1_prime_star", "m2_prime_star"]:
            if p not in list(inferred_params.keys()):
                inferred_params[p] = 0

        return InferredDemography(theta0=inferred_params["theta0"], theta1=inferred_params["theta1"], theta2=inferred_params["theta2"],
                                    theta1_prime=inferred_params["theta1_prime"], theta2_prime=inferred_params["theta2_prime"],
                                      t1=inferred_params["t1"], v=inferred_params["v"], m1_star=inferred_params["m1_star"],
                                        m2_star=inferred_params["m2_star"], m1_prime_star=inferred_params["m1_prime_star"],
                                          m2_prime_star=inferred_params["m2_prime_star"], negll=negll, n_params=n_params)











