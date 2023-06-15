from dismal.likelihood_matrix import LikelihoodMatrix
import numpy as np
import scipy
import tqdm


class DemographicModel:

    def __init__(self, model_name=None, pop_size_change=False, migration=False,
                 mig_rate_change=False, asymmetric_m=False,
                 asymmetric_m_star=False, asymmetric_m_prime_star=False,
                 remove_mig_params=None):
        """TODO: secondary_contact, initial_migration, assymm_sc, assymm_im parameters

        Parameters in next version:
        * demes (specification in Demes format; use given values as initial values)
        * n_stages [2,3]
        * migration [T,F]
        * mig_rate_change
            * mig_stage2
            * mig_stage3
        * asymmetric_mig
            * asymmetric_mig_stage2
            * asymmetric_mig_stage3
        * pop_size_change
        * asymmetric_pop_sizes = T
            * symmetric_popsize_stage2
            * symmetric_popsize_stage3

        Also allow prespecified models using model_name

        """

        self.model_name = model_name

        self.thetas = ["theta0", "theta1", "theta2"]
        self.taus = []
        self.Ms = []

        if pop_size_change or mig_rate_change:
            self.taus.extend(["t1", "v"])
        else:
            self.taus.append("t0")

        if pop_size_change:
            self.thetas.extend(["theta1_prime", "theta2_prime"])

        if migration:
            if not mig_rate_change:
                if asymmetric_m:
                    self.Ms.extend(["M1", "M2"])
                else:
                    self.Ms.append("M")
            else:
                if asymmetric_m_star:
                    self.Ms.extend(["M1_star", "M2_star"])
                else:
                    self.Ms.append("M_star")
                if asymmetric_m_prime_star:
                    self.Ms.extend(["M1_prime_star", "M2_prime_star"])
                else:
                    self.Ms.append("M_prime_star")

        if remove_mig_params is not None:
            for mig_param in remove_mig_params:
                if mig_param in list(self.Ms):
                    self.Ms.remove(mig_param)

        self.model_params = self.thetas + self.taus + self.Ms

    def __str__(self):
        return f"""
        DISMaL DemographicModel object.
        Theta (population size) parameters: {self.thetas}
        Tau (split time) parameters: {self.taus}
        M (migration rate) parameters: {self.Ms}
        """

    def __repr__(self):
        return f"""
        DISMaL DemographicModel object.
        Theta (population size) parameters: {self.thetas}
        Tau (split time) parameters: {self.taus}
        M (migration rate) parameters: {self.Ms}
        """

    @staticmethod
    def composite_likelihood(param_vals, param_names, S, verbose=False):
        params = dict(zip(param_names, param_vals))
        likelihood_matrix = LikelihoodMatrix(params, S)
        negll = np.sum(S * likelihood_matrix.matrix)

        if verbose:
            print(params, negll)

        return negll

    def minimise_neg_likelihood(self, S, thetas_iv, taus_iv, migr_iv, thetas_lb,
                                taus_lb, migr_lb, verbose=False):
        """Find the minimum negative log-likelihood ("fixed" initial values)"""

        initial_thetas = [thetas_iv] * len(self.thetas)
        initial_taus = [taus_iv] * len(self.taus)
        initial_Ms = [migr_iv] * len(self.Ms)
        initial_values = initial_thetas + initial_taus + initial_Ms

        theta_bounds = [(thetas_lb, None)] * len(self.thetas)
        tau_bounds = [(taus_lb, None)] * len(self.taus)
        migr_bounds = [(migr_lb, None)] * len(self.Ms)
        bounds = theta_bounds + tau_bounds + migr_bounds

        for optimisation_algo in ["L-BFGS-B", "Nelder-Mead", "Powell"]:
            optimised = scipy.optimize.minimize(self.composite_likelihood,
                                                x0=initial_values,
                                                method=optimisation_algo,
                                                args=(self.model_params,
                                                      S, verbose),
                                                # options={
                                                #     "ftol": ftol,
                                                #     "eps": eps,
                                                #     "maxfun": maxfun,
                                                #     "maxIs": max_iter
                                                # },
                                                bounds=bounds)
            if optimised.success:
                break
            else:
                print(f"Optimiser {optimisation_algo} failed; trying again")

        if optimised.success:
            inferred_params = dict(zip(self.model_params, optimised.x))
            n_params = len(self.model_params)
            negll = optimised.fun
        else:
            raise RuntimeError(
                "Optimisers L-BFGS-B, Nelder-Mead, and Powell all failed to maximise the likelihood")

        return inferred_params, negll, n_params

    def infer(self, S, n_iterations=5,
              thetas_dist_params=(3, 1),
              taus_dist_params=(3, 1),
              migr_dist_params=(0.3, 1),
              thetas_lb=0.01, migr_lb=0, taus_lb=0,
              verbose=False):
        """Repeats optimisation; samples IVs from gamma distribution

        Args:
            S (_type_): _description_
            verbose (bool, optional): _description_. Defaults to False.
            optimisation_algo (str, optional): _description_. Defaults to "L-BFGS-B".
        """

        # sample initial values from gamma dist
        initial_thetas = scipy.stats.gamma.rvs(
            a=thetas_dist_params[0], scale=thetas_dist_params[1], size=n_iterations)
        initial_taus = scipy.stats.gamma.rvs(
            a=taus_dist_params[0], scale=taus_dist_params[1], size=n_iterations)
        initial_migr = scipy.stats.gamma.rvs(
            a=migr_dist_params[0], scale=migr_dist_params[1], size=n_iterations)

        models = []
        for i in tqdm.tqdm(range(n_iterations)):
            if verbose:
                print(
                    f"Optimisation iteration {i} with initial theta = {initial_thetas[i]}, initial tau = {initial_taus[i]} and initial M = {initial_migr[i]}")
            mod = self.minimise_neg_likelihood(
                S=S, thetas_iv=initial_thetas[i], taus_iv=initial_taus[i],
                migr_iv=initial_migr[i], thetas_lb=thetas_lb, taus_lb=taus_lb, migr_lb=migr_lb, verbose=verbose)
            models.append(mod)

        neglls = [models[i][1] for i in range(len(models))]
        return models[neglls.index(min(neglls))]  # return best model
