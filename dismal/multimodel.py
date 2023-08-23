import numpy as np
from divergencemodel import DivergenceModel


class MultiModel:

    def __init__(self, s1, s2, s3, blocklen,
                 deme1_id, deme2_id,
                 min_epochs=2, max_epochs=3,
                 allow_asymmetric_migration=True):
        
        for s_arr in [s1, s2, s3]:
            assert isinstance(s_arr, np.ndarray), \
                "s1, s2, and s3 must be NumPy arrays where each entry s1[i] corresponds to the count of i segregating sites"
        assert 2 <= min_epochs <= 3, "min_epochs must be either 2 or 3"
        assert 2 <= max_epochs <= 3, "max_epochs must be either 2 or 3"

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3 
        self.blocklen = blocklen
        self.deme1_id = deme1_id
        self.deme2_id = deme2_id
        self.deme_ids = [deme1_id, deme2_id]
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.epoch_range = [min_epochs, max_epochs]
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

        if 2 in self.epoch_range:
            model_specs.append([{"epochs":2, "migration": (False, False)},
                               {"epochs": 2, "migration": (True, False), "asym_migration": (False, False)}])
            if self.allow_asymmetric_migration is True:
                model_specs.append({"epochs":2, "migration": (True, False), "asym_migration": (True, False)})
        
        if 3 in self.epoch_range:
            model_specs.append([{"epochs": 3, "migration": (False, False, False)},
                               {"epochs": 3, "migration": (False, True, False), "asym_migration": (False, False, False)},
                               {"epochs": 3, "migration": (True, False, False), "asym_migration": (False, False, False)},
                               {"epochs": 3, "migration": (True, True, False), "asym_migration": (False, False, False)}])
            if self.allow_asymmetric_migration:
                model_specs.append([{"epochs":3, "migration": (False, True, False), "asym_migration": (False, True, False)},
                                    {"epochs":3, "migration": (True, False, False), "asym_migration": (True, False, False)},
                                    {"epochs":3, "migration": (True, True, False), "asym_migration": (True, True, False)}])
                
        return model_specs
    
    def add_model_spec(self, spec_dict):
        """Add a model specification in the form of a dictionary.

        For each key other than 'epochs', the value must be a tuple of length epochs, i.e. (False, False, True). 
        To control migration direction, provide a tuple of deme IDs for each epoch,
          e.g. (("Human", "Chimp), ("Human", "Chimp"), ("None", "None")) sets unidirectional migration Human -> Chimp for the two epochs of migration.
        Remember that DISMaL models are always specified with the most recent epoch first.

        The following dictionary keys are allowed:
        * epochs (required)
        * migration (required)
        * asym_migration (optional: controls whether asymmetric migration is allowed; default is True if migration is allowed)
        * migration_direction (optional: sets unidirectional migration as above; default is None (bidirectional migration))
        
        """
        self.model_space.append(spec_dict)
    
    def fit_models(self):

        for model_spec in self.model_space:
            mod = DivergenceModel()
            epoch_idxs = range(model_spec["epochs"]-1)

            for epoch_idx in epoch_idxs:
                allow_migration_epoch = model_spec["migration"][epoch_idx]
                assert isinstance(allow_migration_epoch, bool)

                allow_asymmetric_migration_epoch = model_spec["asym_migration"][epoch_idx]
                if allow_asymmetric_migration_epoch is None:
                    allow_asymmetric_migration_epoch = False

                migration_direction_epoch = model_spec["migration_direction"][epoch_idx] # defaults to None
                if migration_direction_epoch is not None:
                    assert migration_direction_epoch[0] in self.deme_ids \
                        and migration_direction_epoch[1] in self.deme_ids
                
                mod.add_epoch(index=epoch_idx,
                              deme_ids = self.deme_ids,
                              migration = allow_migration_epoch,
                              asymmetric_migration = allow_asymmetric_migration_epoch,
                              migration_direction = migration_direction_epoch)
                
            mod.add_epoch(index=model_spec["epochs"]-1, deme_ids=["ancestral"], migration=False)
            mod.fit(self.s1, self.s2, self.s3, self.blocklen)
            self.models.append(mod)




        
        
        








