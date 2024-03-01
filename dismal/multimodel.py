from dismal.demographicmodel import DemographicModel
from dismal import models
import numpy as np
import pandas as pd
import tabulate
from scipy.stats.distributions import chi2

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)

class MultiModel:

    def __init__(self, s1, s2, s3, sampled_deme_names, max_epochs=3):
        """Fit multiple models"""

        self.models = [
            models.iso_two_epoch(sampled_deme_names=sampled_deme_names),
            models.im(sampled_deme_names=sampled_deme_names)]
        
        if max_epochs == 3:
            self.models.extend([
            models.iso_three_epoch(sampled_deme_names=sampled_deme_names),
            models.iim(sampled_deme_names=sampled_deme_names),
            models.secondary_contact(sampled_deme_names=sampled_deme_names),
            models.gim(sampled_deme_names=sampled_deme_names)
        ])
            
        self.fitted_models = self.fit_models(s1, s2, s3)
        self.model_refs = [mod.model_ref for mod in self.fitted_models]
        self.claics = [mod.claic for mod in self.fitted_models]
        self.lnls = [-mod.negll for mod in self.fitted_models]
        self.aics = [mod.aic for mod in self.fitted_models]
        self.n_params = [mod.n_params for mod in self.fitted_models]
        self.models_df = self.results_dataframe()

    def __repr__(self):
        return self.print_results()

        
    def fit_models(self, s1, s2, s3):
        """Fit multiple models"""
        fitted = []
        for mod in self.models:
            fitted.append(mod.fit_model(s1, s2, s3))
        return fitted
    
    def results_dataframe(self):
        df = pd.DataFrame({"model": self.model_refs, 
              "lnL": self.lnls,
              "aic": self.aics,
              "n_params": self.n_params})

        return df.sort_values("aic", ascending=True)


    def print_results(self):
        print(f"""{tabulate.tabulate(self.models_df, 
                                 headers=["Model ID", "lnL", "AIC", "N params"], 
                                 tablefmt="fancy_grid")}""")
        
        return ""
    
    @staticmethod
    def likelihood_ratio_test(null_mod, alt_mod, alpha=0.05, verbose=True):
        """LRT between two fitted models"""

        assert alt_mod.n_params > null_mod.n_params, "Null model must be nested within alternate model (so alt model must have more parameters)"
        lhood_ratio = 2*(alt_mod.negll)-(null_mod.negll)
        deg_free = alt_mod.n_params - null_mod.n_params
        p = chi2.sf(lhood_ratio, deg_free)

        if verbose:
            print(f"""
            Likelihood ratio test
            Null model: {null_mod.model_ref} with {null_mod.n_params} parameters
            Alternate model: {alt_mod.model_ref} with {alt_mod.n_params} parameters

            Degrees of freedom for chi2 test: {deg_free}
            Critical value for chi2 at alpha {alpha}: {chi2.isf(alpha, (alt_mod.n_params - null_mod.n_params))}
            Likelihood ratio: {lhood_ratio}
            alpha = {alpha}
            p = {p}

            """)

        return (lhood_ratio, p)