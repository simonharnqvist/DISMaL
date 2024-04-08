from dismal.demographicmodel import DemographicModel
from dismal import models
import numpy as np
import pandas as pd
import tabulate
from scipy.stats.distributions import chi2
from joblib import Parallel, delayed
import tqdm
import os
import itertools

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)

class MultiModel:

    def __init__(self, s1, s2, s3, sampled_deme_names, max_epochs=3, threads=1):
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
        
        self.threads = threads
        if self.threads == -1:
            self.threads = os.cpu_count()-1
            
        self.fitted_models = self.fit_models(s1, s2, s3, threads=self.threads)
        self.model_refs = [mod.model_ref for mod in self.fitted_models if mod is not None]
        #self.claics = [mod.claic for mod in self.fitted_models]
        self.lnls = [-mod.negll for mod in self.fitted_models if mod is not None]
        self.aics = [mod.aic for mod in self.fitted_models if mod is not None]
        self.n_params = [mod.n_params for mod in self.fitted_models if mod is not None]
        self.models_df = self.results_dataframe()

    def __repr__(self):
        return self.print_results()
    
    def fit_single_model(self, mod, s1, s2, s3):
        try:
            return mod.fit_model(s1, s2, s3)
        except RuntimeError:
            return None

        
    def fit_models(self, s1, s2, s3, threads=1):
        """Fit multiple models"""
        fitted = []
        # for mod in self.models:
        #     fitted.append(mod.fit_model(s1, s2, s3))
        # return fitted
    
        fitted.append(
            list(tqdm.tqdm(Parallel(n_jobs=threads, return_as="generator")(
                delayed(self.fit_single_model)(
                    mod, s1, s2, s3
                    ) for mod in self.models), total=len(self.models))))
        
        return list(itertools.chain(*fitted))
    
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

        assert alt_mod.n_params > null_mod.n_params, "Alt model must have more parameters than null (they must be nested, which is not checked)"
        lnL_null = -null_mod.negll
        lnL_alt = -alt_mod.negll
        lhood_ratio = -2*(lnL_null - lnL_alt)
        deg_free = alt_mod.n_params - null_mod.n_params
        p = chi2.sf(lhood_ratio, deg_free)

        if verbose:
            print(f"""
            Likelihood ratio test
            Null model: {null_mod.model_ref} with {null_mod.n_params} parameters, lnL: {lnL_null}
            Alternate model: {alt_mod.model_ref} with {alt_mod.n_params} parameters, lnL: {lnL_alt}

            Degrees of freedom for chi2 test: {deg_free}
            Critical value for chi2 at alpha {alpha}: {chi2.isf(alpha, (alt_mod.n_params - null_mod.n_params))}
            Log-likelihood ratio: {lhood_ratio}
            alpha = {alpha}
            p = {p}

            """)

        return (lhood_ratio, p)