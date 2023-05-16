from likelihood_matrix import LikelihoodMatrix
import numpy as np
import scipy

class DemographicModel:

    def __init__(self, pop_size_change=False, migration=False, mig_rate_change=False,
                  assymmetric_m=False, assymmetric_m_star=False, assymmetric_m_prime_star=False,
                    remove_mig_params=None, thetas_iv=5, mig_rates_iv=0.3, taus_iv=5,
                      thetas_lb=0.01, mig_rates_lb=0, taus_lb=0):
        
        self.thetas = {"theta0":(thetas_iv, thetas_lb), "theta1":(thetas_iv, thetas_lb), "theta2":(thetas_iv, thetas_lb)}
        self.mig_rates = {}
        self.taus = {}

        if pop_size_change or mig_rate_change:
            self.taus["t1"] = (taus_iv, taus_lb)
            self.taus["v"] = (taus_iv, taus_lb)
        else:
            self.taus["t0"] = (taus_iv, taus_lb)
        
        if pop_size_change:
            self.thetas["theta1_prime"] = (thetas_iv, thetas_lb)
            self.thetas["theta2_prime"] = (thetas_iv, thetas_lb)

        if migration:
            if not mig_rate_change:
                if assymmetric_m:
                    self.mig_rates["M1"] = (mig_rates_iv, mig_rates_lb)  
                    self.mig_rates["M2"] = (mig_rates_iv, mig_rates_lb) 
                else:
                    self.mig_rates["M"] = (mig_rates_iv, mig_rates_lb)
            else:
                if assymmetric_m_star:
                    self.mig_rates["M1_star"] = (mig_rates_iv, mig_rates_lb)
                    self.mig_rates["M2_star"] = (mig_rates_iv, mig_rates_lb)
                else:
                    self.mig_rates["M_star"] = (mig_rates_iv, mig_rates_lb)
                if assymmetric_m_prime_star:
                    self.mig_rates["M1_prime_star"] = (mig_rates_iv, mig_rates_lb)
                    self.mig_rates["M2_prime_star"] = (mig_rates_iv, mig_rates_lb)
                else:
                    self.mig_rates["M_prime_star"] = (mig_rates_iv, mig_rates_lb)
        
        if remove_mig_params is not None:
            for mig_param in remove_mig_params:
                if mig_param in list(self.mig_rates.keys()):
                    self.mig_rates.pop(mig_param)

        self.param_names = list(self.thetas.keys()) + list(self.taus.keys()) + list(self.mig_rates.keys())
        self.initial_values = [i[0] for i in self.thetas.values()] + [i[0] for i in self.taus.values()] + [i[0] for i in self.mig_rates.values()]
        self.lower_bounds = [i[1] for i in self.thetas.values()] + [i[1] for i in self.taus.values()] + [i[1] for i in self.mig_rates.values()]
        print(f"Model parameters: {self.param_names}, Initial values: {self.initial_values} Lower bounds: {self.lower_bounds}")

    @staticmethod
    def composite_likelihood(param_vals, param_names, S, verbose=False):
        params = dict(zip(param_names, param_vals))
        likelihood_matrix = LikelihoodMatrix(params, S)
        negll = np.sum(S * likelihood_matrix.matrix)

        if verbose:
            print(params, negll)

        return negll
    
    def infer(self, S, verbose):

        none_list = [None for i in range(0, len(self.param_names))]
        bounds = list(zip(self.lower_bounds, none_list))

        for optimisation_algo in ["L-BFGS-B", "Nelder-Mead", "Powell"]:
            optimised = scipy.optimize.minimize(self.composite_likelihood, x0=np.array(self.initial_values),
                                                method=optimisation_algo,
                                                args=(self.param_names, S, verbose),
                                                bounds=bounds)
            if optimised.success:
                break
            else:
                f"Optimiser {optimisation_algo} failed"

        if optimised.success:
            inferred_params = dict(zip(self.param_names, optimised.x))
            negll = optimised.fun
        else:
            raise RuntimeError(f"Optimisers L-BFGS-B, Nelder-Mead, and Powell all failed to maximise the likelihood")

        return inferred_params, negll




        

        

        
