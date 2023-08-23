class Epoch:

    def __init__(self, 
                 index,
                 deme_ids,
                 migration,
                 asymmetric_migration=True,
                 migration_direction=None):
        assert isinstance(index, int)
        
        self.index = index
        self.deme_ids = deme_ids

        assert (isinstance(migration, list) and isinstance(migration[0], bool)) \
            or isinstance(migration, bool), "migration argument must be bool or list of bools"
        self.migration = migration

        if self.migration is False:
            self.asymmetric_migration = False
            self.migration_direction = False
        else:
            self.asymmetric_migration = asymmetric_migration
            self.migration_direction = migration_direction
        self.n_m_params = self.n_migration_rates()

    def n_migration_rates(self):

        if self.migration is False:
            return 0
        elif self.asymmetric_migration is False:
            return 1
        elif self.migration_direction is not None:
            return 1
        else:
            return 2