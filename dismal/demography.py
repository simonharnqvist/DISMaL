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
                 set_m1_zero = False, set_m2_zero = False, set_m1_prime_zero = False, set_m2_prime_zero = False, no_migration=False,
                 a_initial=1, b_initial=1, c1_initial=1, c2_initial=1, tau0_initial=1, tau1_initial=2, m1_initial=0, m2_initial=0, 
                 m1_prime_initial=0, m2_prime_initial=0, theta_initial=5, 
                 a_lower=0.001, b_lower=0.001, c1_lower=0.001, c2_lower=0.001, tau0_lower=0.001, tau1_lower=0.001, m1_lower=0, m2_lower=0, 
                 m1_prime_lower=0, m2_prime_lower=0, theta_lower=0.0000001):

        """
        Constructs a Demography() object.
        :param list x1: list of number of segregating sites (nucleotide differences) between loci from population A2
        :param list x2: list of number of segregating sites (nucleotide differences) between loci from population B2
        :param list x3: list of number of segregating sites (nucleotide differences) between loci from different populations
        :param str model_description: description of the model
        :param dict popnames: dictionary of population names
        :param abs_popsize_b1: absolute population size (if known) of population 1 between tau1 and tau0
        :param dict initial_vals: initial values for optimisation
        :param dict lower_bounds: lower bounds for optimisation
        :param dict uppper_bounds: upper bounds for optimisation
        :param bool no_migration: force all migration rates to zero (disallow any migration)
        """
        
        if model.lower() in ["iso", "isolation"]:
            no_migration = True
        elif model.lower() in ["sc", "sec", "secondary_contact"]:
            set_m1_zero = True
            set_m2_zero = True
        elif model.lower() in ["iim", "initial_migration", "am", "ancestral_migration"]:
            set_m1_prime_zero = True
            set_m2_prime_zero = True
        elif model.lower() == "gim":
            pass
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

        initial_vals_model_params = [a_initial, b_initial, c1_initial, c2_initial, tau0_initial, tau1_initial,
                         m1_initial, m2_initial, m1_prime_initial, m2_prime_initial, theta_initial]
        self.initial_vals = utils.model_to_opt_params(initial_vals_model_params)

        lower_bounds_model_params = [a_lower, b_lower, c1_lower, c2_lower, tau0_lower, tau1_lower,
                         m1_lower, m2_lower, m1_prime_lower, m2_prime_lower, theta_lower]
        self.lower_bounds = utils.model_to_opt_params(lower_bounds_model_params)

        # if no migration allowed, set upper bounds for mig params to 0
        upper_migration_params = []
        for migration_parameter in [self.set_m1_zero, self.set_m2_zero, self.set_m1_prime_zero, self.set_m2_prime_zero]:
            if migration_parameter:
                upper_migration_params.append(0)
            else:
                upper_migration_params.append(None)

        self.upper_bounds = [None, None, None, None, None, None, None] + upper_migration_params

        self.model_description = model_description
        self.X = X

    #     self.res = self.infer_parameters()
    #
    # def __repr__(self):
    #     return self.res

    def infer_parameters(self, optimisation_type="local", optimisation_algo='L-BFGS-B'):


        optim_bounds = tuple(zip(self.lower_bounds, self.upper_bounds))

        if optimisation_type == "basinhopping":
            minimizer_kwargs = dict(method=optimisation_algo, args=self.X, bounds=optim_bounds)
            optimised = scipy.optimize.basinhopping(likelihood._composite_neg_ll, x0=np.array(self.initial_vals),
                                                    minimizer_kwargs=minimizer_kwargs)
        elif optimisation_type == "local":
            optimised = scipy.optimize.minimize(likelihood._composite_neg_ll, x0=np.array(self.initial_vals),
                                                method=optimisation_algo, args=self.X, bounds=optim_bounds)
        else:
            raise ValueError("Please specify 'local', 'basinhopping', or 'differential_evolution' optimisation")

        inferred_params = optimised.x
        negll = optimised.fun
        a, b, c1, c2, tau1, tau0, m1, m2, m1_prime, m2_prime, theta = utils.opt_to_model_params(inferred_params)

        n_mig_rates = 4-len([self.set_m1_zero, self.set_m2_zero,
                                             self.set_m1_prime_zero, self.set_m2_prime_zero])
        n_params = len(inferred_params-4) + n_mig_rates

        assert optimised.success, f"Optimisation failed: {optimised.message}"

        return InferredDemography(model_description=self.model_description, a=a,
                                  b=b, c1=c1, c2=c2, tau1=tau1, tau0=tau0, m1=m1, m2=m2, m1_prime=m1_prime,
                                  m2_prime=m2_prime, theta=theta, negll=negll, n_params=n_params)



def compare_four_models(vcf_path, samples_path, true_model):
    s_counts = preprocess.vcf_to_s_count(vcf_path="../test.vcf", samples_path="../samples.txt", n_blocks=64, block_length=10000)
    iso = Demography(population_names=["anc", "b1", "b2", "c1", "c2"], x1=s_counts[0], x2=s_counts[1], x3=s_counts[2], no_migration=True).infer_parameters()
    gim = Demography(population_names=["anc", "b1", "b2", "c1", "c2"], x1=s_counts[0], x2=s_counts[1], x3=s_counts[2], no_migration=False).infer_parameters()
    sec = Demography(population_names=["anc", "b1", "b2", "c1", "c2"], x1=s_counts[0], x2=s_counts[1], x3=s_counts[2], allow_mig_A1B1 = False, allow_mig_B1A1 = False, allow_mig_A2B2 = True, allow_mig_B2A2 = True,).infer_parameters()
    iim = Demography(population_names=["anc", "b1", "b2", "c1", "c2"], x1=s_counts[0], x2=s_counts[1], x3=s_counts[2], allow_mig_A1B1 = True, allow_mig_B1A1 = True, allow_mig_A2B2 = False, allow_mig_B2A2 = False).infer_parameters()
    neglls = {"iso": iso.negll, "iim": iim.negll, "sec":sec.negll, "gim":gim.negll}
    aics = {"iso": iso.aic, "iim": iim.aic, "sec":sec.aic, "gim":gim.aic}

    best_mod_negll = min(neglls, key=neglls.get)
    best_mod_aic = min(aics, key=aics.get)

    return [true_model, best_mod_negll, best_mod_aic]











