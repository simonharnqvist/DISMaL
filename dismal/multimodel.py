import numpy as np
from dismal.divergencemodel import DivergenceModel


class MultiModel:

    def __init__(self,
                 deme1_id, deme2_id,
                 allow_asymmetric_migration=True):

        self.deme1_id = deme1_id
        self.deme2_id = deme2_id
        self.deme_ids = [deme1_id, deme2_id]
        self.n_demes = 2   # for future extension
        self.allow_asymmetric_migration = allow_asymmetric_migration
        self.model_space = self.make_model_space()
        self.models = []
    
    def make_model_space(self):
        """ 
        Create a list of dictionaries, each specifying a model to fit.
        Future TODO: Would be nicer to do this dynamically.
        """
        
        model_specs = []
        model_specs.extend([{"model_ref": "isolation", "deme_ids": self.deme_ids, "epochs": 3, "migration": (False, False, False)},
                            {"model_ref": "iim_symmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (False, True, False), "asym_migration": (False, False, False)},
                            {"model_ref": "secondary_contact_symmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (True, False, False), "asym_migration": (False, False, False)},
                            {"model_ref": "gim_symmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (True, True, False), "asym_migration": (False, False, False)}])
        if self.allow_asymmetric_migration:
            model_specs.extend([{"model_ref": "iim_asymmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (False, True, False), "asym_migration": (False, True, False)},
                                {"model_ref": "secondary_contact_asymmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (True, False, False), "asym_migration": (True, False, False)},
                                {"model_ref": "gim_asymmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (True, True, False), "asym_migration": (True, True, False)}])
                
        return model_specs
    
    def add_model_spec(self, spec_dict):
        """Add a model specification in the form of a dictionary.

        For each key other than 'epochs', the value must be a tuple of length epochs, i.e. (False, False, True). 
        To control migration direction, provide a tuple of deme IDs for each epoch,
          e.g. (("Human", "Chimp), ("Human", "Chimp"), ("None", "None")) sets unidirectional migration Human -> Chimp for the two epochs of migration.
        Remember that DISMaL models are always specified with the most recent epoch first.

        The following dictionary keys are allowed:
        * model_ref (required, but does not need to be unique)
        * epochs (required)
        * deme_ids (required)
        * migration (required)
        * asym_migration (optional: controls whether asymmetric migration is allowed; default is True if migration is allowed)
        * migration_direction (optional: sets unidirectional migration as above; default is None (bidirectional migration))
        
        """
        self.model_space.append(spec_dict)

    def fit_models(self, s1, s2, s3, blocklen, verbose=False):

        for s_arr in [s1, s2, s3]:
            assert isinstance(s_arr, np.ndarray), \
                "s1, s2, and s3 must be NumPy arrays where each entry s1[i] corresponds to the count of i segregating sites"

        for dict_spec in self.model_space:
            mod = DivergenceModel.from_dict_spec(dict_spec)
            mod.fit(s1, s2, s3, blocklen)

            if verbose:
                print(mod.res)
            self.models.append(mod)


        
        
        








