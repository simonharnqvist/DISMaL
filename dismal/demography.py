
class Epoch:

    def __init__(self, 
                 deme_ids,
                 migration,
                 asymmetric_migration=True,
                 migration_direction=None,
                 migration_rates = None,
                 thetas = None,
                 start_time=None,
                 end_time=None):

        self.deme_ids = deme_ids
        self.n_thetas = len(deme_ids)

        assert (isinstance(migration, list) and isinstance(migration[0], bool)) \
            or isinstance(migration, bool), "migration argument must be bool or list of bools"
        self.migration = migration

        if self.migration is False:
            self.asymmetric_migration = False
            self.migration_direction = None
        else:
            self.asymmetric_migration = asymmetric_migration
            self.migration_direction = migration_direction
        self.n_m_params = self._n_migration_rates()

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
    
