import demes
import numpy as np
import demesdraw
import msprime
from dismal.simulate import make_treeseqs, generate_seg_sites_distr

class DemesRepresentation:

    def __init__(self, mod, mutation_rate, blocklen):
        """Demes format representation of DISMaL model"""

        self.mod = mod
        self.blocklen = blocklen
        self.mutation_rate = mutation_rate
        self.Ne_scaler = self.theta_to_Ne(mod.epochs[1].thetas[0], 
                                          blocklen=self.blocklen, mu=self.mutation_rate)

        self.builder = self.generate_builder()
        self.graph = self.builder.resolve()
        self.drawing = demesdraw.tubes(self.graph, labels="xticks")


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
                twoNe = self.theta_to_Ne(epoch.thetas[deme_idx], blocklen=self.blocklen, mu=self.mutation_rate)
                end_t_gen = self.t_to_generations(epoch.start_time, (self.Ne_scaler/2))

                if epoch_idx == 0:
                    builder.add_deme(deme_id, 
                        epochs=[dict(end_time=end_t_gen, start_size=twoNe)])
                elif epoch_idx == 1:
                    builder.add_deme(deme_id, 
                        ancestors=[reversed_epochs[0].deme_ids[0]], 
                        epochs=[dict(end_time=end_t_gen, start_size=twoNe)])
                else:
                    builder.add_deme(deme_id, 
                        ancestors=[reversed_epochs[epoch_idx-1].deme_ids[deme_idx]], 
                        epochs=[dict(end_time=end_t_gen, start_size=twoNe)])

            if epoch.migration_rates is not None:
                ms = self.M_to_m(epoch.migration_rates, self.Ne_scaler/2)
                builder.add_migration(source=epoch.deme_ids[0], dest=epoch.deme_ids[1], rate=ms[0])
                builder.add_migration(source=epoch.deme_ids[1], dest=epoch.deme_ids[0], rate=ms[1])

        return builder

        
    def generate_builder(self):
        builder = demes.Builder(
            time_units="generations",
            generation_time=1)
        
        builder = self.add_epochs(builder)
        return builder
    

    def as_msprime_demography(self):
        return msprime.Demography.from_demes(self.graph)


    def simulate_seg_sites(self, recombination_rate=None, num_blocks=20_000):
        """Create segregating sites distributions using coalescent simulations"""
        (treeseqs_state1,
        treeseqs_state2,
        treeseqs_state3) = [make_treeseqs(demography = self.as_msprime_demography(), 
                                       state = state, 
                                       blocklen=self.blocklen, 
                                       recombination_rate=recombination_rate, 
                                       num_blocks=num_blocks) for state in [1, 2, 3]]
        
        s1, s2, s3 = [generate_seg_sites_distr(treeseqs, 
                                               mutation_rate=self.mutation_rate, 
                                               infinite_sites=True) 
                                               for treeseqs in
                                               [treeseqs_state1, treeseqs_state2, treeseqs_state3]]
        
        return s1, s2, s3
        


