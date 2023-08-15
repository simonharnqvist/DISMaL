import numpy as np
import itertools


class Population:

    def __init__(self, id, epoch_id, population_name=None):
        assert isinstance(id, int)
        assert isinstance(epoch_id, int)

        self.id = id
        self.population_name = population_name
        self.epoch_id = epoch_id

class Epoch:

    def __init__(self, id, allow_migration, symmetric_migration=True, migration_direction=None,
                  start_time=None, end_time=None, populations=None):
        assert isinstance(id, int)
        
        self.id = id

        assert (isinstance(allow_migration, list) and isinstance(allow_migration[0], bool)) or isinstance(allow_migration, bool), "migration argument must be bool or list of bools"
        self.allow_migration = allow_migration
        self.symmetric_migration = symmetric_migration
        self.migration_direction = migration_direction

        self.start_time = start_time
        self.end_time = end_time

        self.populations = populations
        
        self.n_M_params = self.n_migration_rates()

    def n_migration_rates(self):

        if self.allow_migration is None:
            return 0
        elif self.symmetric_migration is True:
            return 1
        elif self.migration_direction is not None:
            return 1
        else:
            return 2