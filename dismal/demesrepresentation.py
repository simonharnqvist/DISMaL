import demes
import numpy as np
import demesdraw
import msprime


class DemesRepresentation:

    def __init__(self, mod, mutation_rate, blocklen, log_time=True):
        """Demes format representation of DISMaL model"""

        self.mod = mod
        self.blocklen = blocklen
        self.mutation_rate = mutation_rate
        self.Ne_scaler = self.theta_to_Ne(mod.epochs[1].thetas[0],
                                          blocklen=self.blocklen, mu=self.mutation_rate)

        self.builder = self.generate_builder()
        self.graph = self.builder.resolve()
        self.drawing = demesdraw.tubes(
            self.graph, labels="xticks", log_time=log_time)

    @staticmethod
    def theta_to_Ne(block_theta, blocklen, mu):
        """Convert theta estimate to Ne"""
        return (block_theta/blocklen)/(4*mu)

    @staticmethod
    def t_to_generations(t, Ne):
        """Convert time from Ne generations to generations"""
        return t * 4 * Ne

    @staticmethod
    def M_to_m(M, Ne):
        return np.array(M)/(4*Ne)

    def add_epochs(self, builder):
        reversed_epochs = list(reversed(self.mod.epochs))

        for epoch_idx, epoch in enumerate(reversed_epochs):
            for deme_idx, deme_id in enumerate(epoch.deme_ids):
                twoNe = 2 * \
                    self.theta_to_Ne(
                        epoch.thetas[deme_idx], blocklen=self.blocklen, mu=self.mutation_rate)
                end_t_gen = self.t_to_generations(
                    epoch.start_time, (self.Ne_scaler/2))

                if epoch_idx == 0:
                    builder.add_deme(deme_id,
                                     epochs=[dict(end_time=end_t_gen, start_size=twoNe)])
                elif epoch_idx == 1:
                    builder.add_deme(deme_id,
                                     ancestors=[
                                         reversed_epochs[0].deme_ids[0]],
                                     epochs=[dict(end_time=end_t_gen, start_size=twoNe)])
                else:
                    builder.add_deme(deme_id,
                                     ancestors=[
                                         reversed_epochs[epoch_idx-1].deme_ids[deme_idx]],
                                     epochs=[dict(end_time=end_t_gen, start_size=twoNe)])

            if epoch.migration_rates is not None:
                ms = self.M_to_m(epoch.migration_rates, self.Ne_scaler/2)
                builder.add_migration(
                    source=epoch.deme_ids[0], dest=epoch.deme_ids[1], rate=ms[0])
                builder.add_migration(
                    source=epoch.deme_ids[1], dest=epoch.deme_ids[0], rate=ms[1])

        return builder

    def generate_builder(self):
        builder = demes.Builder(
            time_units="generations",
            generation_time=1)

        builder = self.add_epochs(builder)
        return builder

    def as_msprime_demography(self):
        return msprime.Demography.from_demes(self.graph)
