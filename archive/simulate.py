import msprime
from collections import Counter
import numpy as np

def make_treeseqs(demography, state, blocklen, recombination_rate=0, num_blocks=20_000, ploidy=1):
    if state == 1: 
        n_samps = [2, 0]
    elif state == 2: 
        n_samps = [0, 2]
    elif state == 3: 
        n_samps = [1, 1]

    samples = {demography.populations[-2].name: n_samps[0],
                 demography.populations[-1].name: n_samps[1] }
    print(samples)

    ts_gen = msprime.sim_ancestry(
        samples=samples,
                demography = demography,
                sequence_length = blocklen, 
                recombination_rate = recombination_rate,
                ploidy=ploidy, 
                num_replicates=num_blocks)
    return ts_gen

def generate_seg_sites_distr(treeseqs, mutation_rate, infinite_sites=True):
    discrete_genome = not infinite_sites
    s_counter = Counter([
        msprime.sim_mutations(ts, 
                              rate=mutation_rate, 
                              discrete_genome=discrete_genome)\
                                .segregating_sites(span_normalise=False) 
                                for ts in treeseqs])
    s_max = max(s_counter.keys()) + 1
    s_arr = np.zeros(int(s_max))
    s_arr[np.array(list(s_counter.keys())).astype(int)] = list(s_counter.values())

    return s_arr

