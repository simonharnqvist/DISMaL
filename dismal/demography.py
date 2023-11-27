import secrets

class Epoch:

    def __init__(self, 
                 migration,
                 n_demes=2,
                 deme_ids = None,
                 asymmetric_migration=True,
                 migration_direction=None,
                 migration_rates = None,
                 thetas = None,
                 start_time=None,
                 end_time=None):
        
        assert n_demes in [1,2], "n_demes must be either 1 or 2; multideme epochs not supported"
        self.n_demes = n_demes

        if migration_direction is not None:
            assert deme_ids is not None, "Demes must have unique IDs if migration_direction=True"

        self.deme_ids = deme_ids
        if self.deme_ids is None:
            # if names are not provided, assign unique ID
            pop1_name, pop2_name = f"pop1_{secrets.token_hex(5)}", f"pop2_{secrets.token_hex(5)}"
            self.deme_ids = (pop1_name, pop2_name)[0:n_demes]
        else:
            assert len(deme_ids) == self.n_demes, f"Got {len(deme_ids)} deme IDs, but n_demes={self.n_demes}"

        self.n_thetas = self.n_demes

        assert (isinstance(migration, list) and isinstance(migration[0], bool)) \
            or isinstance(migration, bool), "migration argument must be bool or list of bools"
        self.migration = migration

        if self.migration is False:
            self.asymmetric_migration = False
            self.migration_direction = None
        else:
            self.asymmetric_migration = asymmetric_migration
            self.migration_direction = migration_direction
        self.n_mig_params = self._n_migration_rates()

        self.migration_sources, self.migration_targets = self._get_migration_directions()
        
        self.migration_rates = migration_rates
        self.thetas = thetas
        self.start_time = start_time
        self.end_time = end_time

        if self.start_time is not None and self.end_time is not None:
            assert self.end_time > self.start_time

    def _n_migration_rates(self):

        if self.migration is False:
            return 0
        elif self.asymmetric_migration is False:
            return 1
        elif self.migration_direction is not None:
            return 1
        else:
            return 2

    def _get_migration_directions(self):
        if self.migration is False:
            migration_sources = None
            migration_targets = None
        elif self.migration_direction is not None:
            migration_sources = self.migration_direction[0]
            migration_targets = self.migration_direction[1]
        else:
            migration_sources = self.deme_ids
            migration_targets = self.deme_ids

        return migration_sources, migration_targets
    
