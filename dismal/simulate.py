import msprime
import numpy as np
import random
import itertools
from collections import Counter


class DemographySimulation:

    def __init__(self, block_thetas, epoch_durations, migration_rates,
                 blocklen=500, mutation_rate=1e-9, blocks_per_state=20_000, recombination_rate=0):
        """Msprime simulation

        Args:
            block_theta (iterable): Deme sizes in number of haploid individuals, in DISMaL order (backwards in time)
            epoch_durations (iterable): Durations of epoch 0 and 1 in 2Ne generations
            migration_rates (iterable): Migration rates (big M/number of migrant genomes per generation
            blocklen (int): Length of simulated blocks
            mutation_rate (float): Mutation rate per base
            blocks_per_state (int, optional): Number of block simulations per simulation. Defaults to 20_000.
            recombination_rate (int, optional): Recombination rate per base per generation. Defaults to 0.
        """

        # original parameters
        self.block_thetas = block_thetas
        self.epoch_durations = epoch_durations
        self.blocklen = blocklen
        self.mutation_rate = mutation_rate
        self.blocks_per_state = blocks_per_state
        self.recombination_rate = recombination_rate
        self.deme_ids = ["pop1", "pop2", "pop1_anc", "pop2_anc", "ancestral"]

        # converted parameters for msprime
        self.site_theta = np.array(self.block_thetas)/self.blocklen
        self.deme_sizes = np.array(self.site_theta)/(4*self.mutation_rate)
        self.migration_rates_fraction = np.array(migration_rates)/(self.deme_sizes[0:4])
        self.epoch_durations_generations = np.array(self.epoch_durations) * self.deme_sizes[2]
        self.split_times_generations = self.epoch_durations_generations[0], np.sum(self.epoch_durations_generations)

        self.demography = self.create_demography()
        self.tree_sequences = self.create_treesequences()
        self.s1_counts, self.s2_counts, self.s3_counts = [self.add_mutations(ts) for ts in self.tree_sequences]
        self.s1, self.s2, self.s3 = self.seg_sites_distr()


    def create_demography(self):
        demography = msprime.Demography()

        for deme_idx, deme_id in enumerate(self.deme_ids):
            demography.add_population(name=deme_id, initial_size=self.deme_sizes[deme_idx])

        demography.add_population_split(self.split_times_generations[0], 
                                        derived=[self.deme_ids[0]], 
                                        ancestral=self.deme_ids[2])
        demography.add_population_split(self.split_times_generations[0], 
                                        derived=[self.deme_ids[1]], 
                                        ancestral=self.deme_ids[3])
        demography.add_population_split(self.split_times_generations[1], 
                                        derived=[self.deme_ids[2], 
                                                 self.deme_ids[3]], 
                                                 ancestral=self.deme_ids[4])

        demography.set_migration_rate(self.deme_ids[1], self.deme_ids[0], self.migration_rates_fraction[0])
        demography.set_migration_rate(self.deme_ids[0], self.deme_ids[1], self.migration_rates_fraction[1])
        demography.set_migration_rate(self.deme_ids[3], self.deme_ids[2], self.migration_rates_fraction[2])
        demography.set_migration_rate(self.deme_ids[2], self.deme_ids[3], self.migration_rates_fraction[3])

        demography.sort_events()

        return demography


    def create_treesequences(self):
        ts_state1 = msprime.sim_ancestry(samples={self.deme_ids[0]: 2,
                                                  self.deme_ids[1]: 0},
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen,
                                                  recombination_rate=self.recombination_rate,
                                                  num_replicates=self.blocks_per_state, 
                                                  ploidy=2)
        ts_state2 = msprime.sim_ancestry(samples={self.deme_ids[0]: 0,
                                                  self.deme_ids[1]: 2}, 
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen, 
                                                  recombination_rate=self.recombination_rate,
                                                  num_replicates=self.blocks_per_state, 
                                                  ploidy=2)
        ts_state3 = msprime.sim_ancestry(samples={self.deme_ids[0]: 1, 
                                                  self.deme_ids[1]: 1}, 
                                                  demography=self.demography,
                                                  sequence_length=self.blocklen, 
                                                  recombination_rate=self.recombination_rate,
                                                  num_replicates=self.blocks_per_state, ploidy=2)
        
        return ts_state1, ts_state2, ts_state3


    def add_mutations(self, ts):
        ts_muts = [msprime.sim_mutations(treeseq, rate=self.mutation_rate, discrete_genome=False) for treeseq in ts]
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
    def M2m(Ms, twoNs):
        """Convert big M (number of migrants) to small m (migrant fraction of population)"""
        Ms = np.array(Ms)
        twoNs = np.array(twoNs)

        return Ms/(2*twoNs)


    @staticmethod
    def m2M(ms, twoNs):
        """Convert small m (migrant fraction of population) to big M (number of migrants)"""
        ms = np.array(ms)
        twoNs = np.array(twoNs)

        return 2 * twoNs * ms