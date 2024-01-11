from dismal.demographicmodel import DemographicModel
from dismal import models
import numpy as np
import pandas as pd
import tabulate

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        self.neglls = [mod.negll for mod in self.fitted_models]
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
              "claic": self.claics,
              "negll": self.neglls,
              "n_params": self.n_params})
        
        df["dclaic"] = df["claic"] - min(df["claic"])

        return df.sort_values("dclaic")
    
    def print_results(self):
        print(f"""{tabulate.tabulate(self.models_df, 
                                 headers=["Model ID", "CL AIC", "-lnL", "N params", "Δ CL AIC"], 
                                 tablefmt="fancy_grid")}""")
        
        return ""