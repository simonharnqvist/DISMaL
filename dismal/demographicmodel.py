import numpy as np
import scipy
from dismal.demography import Epoch
from dismal.model_instance import ModelInstance

class DemographicModel:

    def __init__(self, model_ref=None):
        """Represent two (potentially) diverging demes.

        Args:
            model_ref (str, optional): Model reference. Defaults to None.
        """

        self.epochs = None
        self.n_theta_params = 0
        self.n_epoch_durations = -1 # 2 epochs -> 1 duration; 3 epochs -> 2 durations
        self.n_mig_params = 0

        self.negll = None
        self.claic = None


    def add_epoch(self,
                  deme_ids,
                  migration,
                  asymmetric_migration=True,
                  migration_direction=None):
        """Add epoch (period of isolation/gene flow).

        Args:
            deme_ids (iterable): Tuples of deme IDs per epoch, specified backwards in time,
              e.g. [("human", "chimp), ("human", "chimp"), ("ancestor",)]
            migration (bool): Allow migration.
            asymmetric_migration (bool, optional): Allow asymmetric migration. Defaults to True.
            migration_direction (iterable, optional): Tuples specifying migration direction
              per epoch with format (source, destination). Defaults to None.
        """
        
        epoch = Epoch(
            deme_ids=deme_ids,
            migration=migration,
            asymmetric_migration=asymmetric_migration,
            migration_direction=migration_direction
        )

        self.epochs.append(epoch)
        self.deme_ids.append(deme_ids)
        self.n_theta_params = self.n_theta_params + len(deme_ids)
        self.n_epoch_durations = self.n_epoch_durations + 1
        self.n_mig_params = self.n_mig_params + epoch.n_mig_params

    
    def _get_initial_values(self, theta_iv=1, epoch_duration_iv=1, mig_iv=0):
        """Generate list of initial values for parameter estimation"""

        assert len(self.epochs) > 1, "Number of epochs must be at least two; add epochs with add_epochs() method"
        
        thetas_iv = [theta_iv] * self.n_theta_params
        epoch_duration_iv = [epoch_duration_iv] * self.n_epoch_durations
        mig_iv = [mig_iv] * self.n_mig_params
        return thetas_iv + epoch_duration_iv + mig_iv
    

    def _get_bounds(self, theta_bounds=(0.1, None), 
                    epoch_duration_bounds=(0.01, None), 
                    mig_bounds=(None, None)):
        """Generate list of (lower, upper) bound tuples for parameter estimation"""
        thetas_bounds = [theta_bounds] * self.n_theta_params
        epoch_durations_bounds = [epoch_duration_bounds] * self.n_epoch_durations
        migs_bounds = [mig_bounds] * self.n_mig_params
        return thetas_bounds + epoch_durations_bounds + migs_bounds
        
    
    def _negll_wrapper(self, parameter_values, s1, s2, s3):
        """Calculate negative composite log-likelihood"""
        mod = ModelInstance(parameter_values, self.epochs)
        return mod.neg_composite_log_likelihood(s1, s2, s3)

    
    def fit_model(self, s1, s2, s3, initial_values=None, bounds=None, optimisers=None):
        """Fit model by minimising negative log-likelihood; return ModelInstance with optimised parameters"""

        if initial_values is None:
            initial_values = self._get_initial_values()
        if bounds is None:
            bounds = self._get_bounds()
        
        if optimisers is None:
            opt_algos = ["L-BFGS-B", "Nelder-Mead", "Powell"]
        else:
            opt_algos = list(optimisers)
        
        for algo_idx, algo in enumerate(opt_algos):
            optimised = scipy.optimize.minimize(self._negll_wrapper,
                                                x0=initial_values,
                                                method=algo,
                                                args=(s1, s2, s3),
                                                bounds=bounds)
            if optimised.success:
                self.optimiser = algo
                break
            elif algo_idx < len(opt_algos)-1:
                print(f"Optimiser {algo} failed; trying {opt_algos[algo_idx+1]}")
            else:
                raise RuntimeError(
                    f"Optimisers {opt_algos} all failed to maximise the likelihood")

        assert optimised.success, f"Optimisers {opt_algos} all failed to maximise the likelihood"

        return ModelInstance(optimised.x, self.epochs)
    
