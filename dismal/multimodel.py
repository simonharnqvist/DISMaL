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

    def __init__(self, s1: list[int], s2: list[int], s3: list[int],
                 sampled_deme_names: tuple[str], max_epochs: int = 3,
                 threads: int = 1) -> None:
        """Class to fit and represent multiple models fitted on the same data.

        :param s1: Distribution of segregating sites within population 1.
        :type s1: list[int]
        :param s2: Distribution of segregating sites within population 2.
        :type s2: list[int]
        :param s3: Distribution of segregating sites between populations.
        :type s3: list[int]
        :param sampled_deme_names: Names of sampled (current) populations.
        :type sampled_deme_names: tuple[str]
        :param max_epochs: Maximum number of epochs (including ancestral population) to consider, defaults to 3
        :type max_epochs: int, optional
        :param threads: Number of threads to use; 0 = all, -1 = all but one, etc, defaults to 1
        :type threads: int, optional
        """

        self.models = [
            models.iso_two_epoch(sampled_deme_names=sampled_deme_names),
            models.im(sampled_deme_names=sampled_deme_names)]

        if max_epochs == 3:
            self.models.extend([
                models.iso_three_epoch(sampled_deme_names=sampled_deme_names),
                models.iim(sampled_deme_names=sampled_deme_names),
                models.secondary_contact(
                    sampled_deme_names=sampled_deme_names),
                models.gim(sampled_deme_names=sampled_deme_names)
            ])

        self.threads = threads
        if self.threads < 1:
            self.threads = os.cpu_count()+self.threads

        self.fitted_models = self._fit_models(s1, s2, s3, threads=self.threads)
        self.model_refs = [
            mod.model_ref for mod in self.fitted_models if mod is not None]
        # self.claics = [mod.claic for mod in self.fitted_models]
        self.lnls = [-mod.negll for mod in self.fitted_models if mod is not None]
        self.aics = [mod.aic for mod in self.fitted_models if mod is not None]
        self.n_params = [
            mod.n_params for mod in self.fitted_models if mod is not None]
        self.models_df = self._results_dataframe()

    def __repr__(self):
        return self._print_results()

    def _fit_single_model(self, mod, s1, s2, s3):
        try:
            return mod.fit_model(s1, s2, s3)
        except RuntimeError:
            return None

    def _fit_models(self, s1, s2, s3, threads: int = 1) -> list:
        fitted = []

        fitted.append(
            list(tqdm.tqdm(Parallel(n_jobs=threads, return_as="generator")(
                delayed(self._fit_single_model)(
                    mod, s1, s2, s3
                ) for mod in self.models), total=len(self.models))))

        return list(itertools.chain(*fitted))

    def _results_dataframe(self):
        df = pd.DataFrame({"model": self.model_refs,
                           "lnL": self.lnls,
                           "aic": self.aics,
                           "n_params": self.n_params})

        return df.sort_values("aic", ascending=True)

    def _print_results(self):
        print(f"""{tabulate.tabulate(self.models_df, 
                                 headers=["Model ID", "lnL", "AIC", "N params"], 
                                 tablefmt="fancy_grid")}""")

        return ""

    @staticmethod
    def likelihood_ratio_test(null_mod: DemographicModel,
                              alt_mod: DemographicModel, alpha: float = 0.05,
                              verbose: bool = True) -> tuple[float]:
        """Likelihood ratio test between two fitted models. Does not adjust for composite likelihood non-independence.

        :param null_mod: Null model - must be nested within alternate model.
        :type null_mod: DemographicModel
        :param alt_mod: Model to test against null.
        :type alt_mod: DemographicModel
        :param alpha: Alpha parameter for significance, defaults to 0.05
        :type alpha: float, optional
        :param verbose: Verbosity, defaults to True
        :type verbose: bool, optional
        :return: (likelihood ratio, p-value)
        :rtype: tuple[float]
        """

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
