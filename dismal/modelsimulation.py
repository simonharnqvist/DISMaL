from dismal import demography
import msprime
from collections import Counter
import numpy as np

class ModelSimulation:

    def __init__(self, modinst, mutation_rate, blocklen, recombination_rate=0, blocks_per_state=20_000):
        self.modinst = modinst
        self.mu = mutation_rate
        self.rho = recombination_rate
        self.blocklen = blocklen
        self.blocks_per_state = blocks_per_state

        self.ref_Ne = (self.modinst.thetas[-3]/self.blocklen)/(4*self.mu)
        self.Ne_epoch_durations = np.array(self.modinst.epoch_durations) * 2 * self.ref_Ne
        self.Ne_split_times = [np.sum(self.Ne_epoch_durations[0:i]) 
                               for i in range(1, len(self.Ne_epoch_durations)+1)]

        self.demography = self.generate_demography()
        self.sampled_deme_ids = [self.demography.populations[-2].name, self.demography.populations[-1].name]

        self.tree_sequences = self.create_treesequences()
        self.s1_counts, self.s2_counts, self.s3_counts = [self.simulate_seg_sites(ts) for ts in self.tree_sequences]
        self.s1, self.s2, self.s3 = self.seg_sites_distr()

    def add_populations(self, demography):
        for epoch_idx, epoch in enumerate(self.modinst.epochs):
            thetas = epoch.thetas
            n_individs = [(theta/self.blocklen)/(4*self.mu) for theta in thetas]
            [demography.add_population(name=deme_id, initial_size=n) for deme_id, n in list(zip(epoch.deme_ids, n_individs))]

    def add_population_splits(self, demography):
        for epoch_idx, epoch in enumerate(self.modinst.epochs[0:-1]):
            split_time = self.Ne_split_times[epoch_idx]
            if len(list(self.modinst.epochs[epoch_idx+1].deme_ids)) == 1:
                ancestral = list(self.modinst.epochs[epoch_idx+1].deme_ids) * 2
            else:
                ancestral = list(self.modinst.epochs[epoch_idx+1].deme_ids)

            [demography.add_population_split(time=split_time,
                                        derived=[list(epoch.deme_ids)[i]],
                                        ancestral=ancestral[i]) for i in range(len(epoch.deme_ids))]
            
    def add_migration(self, demography):
        for epoch_idx, epoch in enumerate(self.modinst.epochs[0:-1]):
            if epoch.migration is True:
                if epoch.n_mig_params == 2:
                    ms = [self.convert_M_to_m(M, block_theta, self.mu, self.blocklen) 
                          for (M, block_theta) in zip(epoch.migration_rates, epoch.thetas)]
                    
                    demography.set_migration_rate(source=epoch.deme_ids[0], dest=epoch.deme_ids[1],
                                               rate=ms[0])
                    demography.set_migration_rate(source=epoch.deme_ids[1], dest=epoch.deme_ids[0],
                                               rate=ms[1])

                elif epoch.migration_direction is not None:
                    source_idx = np.where([epoch.migration_direction[0] == epoch.deme_ids])
                    M = epoch.migration_rates[0]
                    m = self.convert_M_to_m(M, block_theta=epoch.thetas[source_idx], 
                                              mu=self.mu, blocklen=self.blocklen)
                    demography.set_migration_rate(source=epoch.migration_direction[0],
                                                  target=epoch.migration_direction[1],
                                                  rate=ms)

                elif epoch.asymmetric_migration is True:
                    theta = epoch.thetas[0]
                    M = epoch.migration_rates[0]
                    m = self.convert_M_to_m(M, theta, self.mu, self.blocklen)
                    demography.set_symmetric_migration_rate(populations=epoch.deme_ids, rate=ms)

                else:
                    assert epoch.n_mig_params in [0, 1, 2], "Number of mig params must be 0, 1, or 2"
                    raise ValueError("Invalid combination of migration parameters")

    def generate_demography(self):
        demography = msprime.Demography()
        self.add_populations(demography)
        self.add_population_splits(demography)
        if self.modinst.migration_rates is not None:
            self.add_migration(demography)
        demography.sort_events()

        return demography
    
    def create_treesequences(self):
        ts_state1 = msprime.sim_ancestry(samples={self.sampled_deme_ids[0]: 2,
                                                  self.sampled_deme_ids[1]: 0},
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen,
                                                  recombination_rate=self.rho,
                                                  num_replicates=self.blocks_per_state, 
                                                  ploidy=2)
        ts_state2 = msprime.sim_ancestry(samples={self.sampled_deme_ids[0]: 0,
                                                  self.sampled_deme_ids[1]: 2}, 
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen, 
                                                  recombination_rate=self.rho,
                                                  num_replicates=self.blocks_per_state, 
                                                  ploidy=2)
        ts_state3 = msprime.sim_ancestry(samples={self.sampled_deme_ids[0]: 1, 
                                                  self.sampled_deme_ids[1]: 1}, 
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen, 
                                                  recombination_rate=self.rho,
                                                  num_replicates=self.blocks_per_state, ploidy=2)
        
        return ts_state1, ts_state2, ts_state3


    def simulate_seg_sites(self, ts):
        ts_muts = [msprime.sim_mutations(treeseq, rate=self.mu, discrete_genome=False) for treeseq in ts]
        s = np.array([ts_mut.divergence(sample_sets=[[0], [2]], span_normalise=False) for ts_mut in ts_muts])
            
        return s
    
    def seg_sites_distr(self):
        seg_sites_spec = []

        s1_counter, s2_counter, s3_counter = [Counter(counts) 
                                              for counts in [self.s1_counts, self.s2_counts, self.s3_counts]]

        for s_counter in [s1_counter, s2_counter, s3_counter]:
            s_max = max(s_counter.keys()) + 1
            s_arr = np.zeros(int(s_max))
            s_arr[np.array(list(s_counter.keys())).astype(int)] = list(s_counter.values())
            seg_sites_spec.append(s_arr)

        return seg_sites_spec

    @staticmethod
    def convert_M_to_m(M, block_theta, mu, blocklen):
        site_theta = block_theta/blocklen
        m = M/(site_theta/mu)
        return m
