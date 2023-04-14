from collections import Counter
import numpy as np
import utils
import itertools
import allel
from collections import defaultdict
import random
import functools, operator, collections
import pandas as pd
import glob
import tqdm

# Workflow: 1) read VCF; 2) split into blocks; 3) select 1 hap per individual; 4) calculate s

# TODO:
# Unphased data (random phasing)

def read_r_simulation(filepath):
        x = open(filepath, 'r').read().replace("\n", " ")
        return [int(i) for i in x.replace("\n", " ").split()]

class NestedDefaultDict(defaultdict):
    """From https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict"""
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return str(dict(self))

class BlocksDictionary(defaultdict):
    """This is a mess"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create(self, vcf_path, make_blocks=False, block_length=None, n_blocks=None):
        blocks_dictionary = NestedDefaultDict()

        vcf_dict = dict(allel.read_vcf(vcf_path, fields=["samples", "calldata/GT", "variants/CHROM", "variants/POS"]))
        chromosomes = vcf_dict["variants/CHROM"]
        samples = vcf_dict["samples"]
        gt = vcf_dict["calldata/GT"]
        pos = vcf_dict["variants/POS"]

        if make_blocks:
            assert block_length is not None
            assert n_blocks is not None

            for chr in list(set(chromosomes)):
                block_idx = self.make_blocks_indices_chr(chr=chr, chromosome_idx=chromosomes, pos=pos, block_length=block_length, n_blocks=n_blocks)
                for (start_idx, end_idx) in block_idx:
                    for i in range(0, len(samples)):
                        for hap in [0,1]:
                            blocks_dictionary[chr][f"{chr}:{start_idx}-{end_idx}"][samples[i]][f"hap_{hap+1}"] = gt[start_idx:end_idx, i][:,hap]
        else: # treat VCF as single block
            for i in range(0, len(samples)):
                for hap in [0,1]:
                    blocks_dictionary["1"]["1"][samples[i]][f"hap_{hap+1}"] = gt[:, i][:,hap]

        return blocks_dictionary
    
    @staticmethod
    def make_blocks_indices_chr(chr, chromosome_idx, pos, block_length, n_blocks):
        """Make blocks for one chromosome"""

        chr_indices = np.where(chromosome_idx == chr)

        min_pos = min(pos[chr_indices])
        max_pos = max(pos[chr_indices])
        chr_len = max_pos-min_pos
  
        n_all_blocks = int(chr_len/block_length)
        assert n_all_blocks > n_blocks, "n_blocks is too high"
    
        block_start_positions = np.array([i*block_length for i in range(0, n_all_blocks)]) + min_pos
        block_end_positions = block_start_positions + block_length
        blocks_idx = list(zip(block_start_positions, block_end_positions))

        sampled_blocks_idx = random.sample(blocks_idx, int(n_blocks))

        return sampled_blocks_idx
    
def block_haplotypes(blockdict, chr, block, use_both_haps=False):
    block = blockdict[chr][block]

    if use_both_haps:
        raise NotImplementedError("Using both haplotypes per block is not yet implemented")
    
    haplotypes = []
    for sample in block.keys():
        i = random.randint(0,1)
        haplotypes.append(block[sample][f"hap_{i+1}"])

    return haplotypes
    
def block_haplotypes_per_population(blockdict, chr, block, population, samples_to_pop_map, use_both_haps=False):
    block = blockdict[chr][block]

    if use_both_haps:
        raise NotImplementedError("Using both haplotypes per block is not yet implemented")
    
    haplotypes = []

    for sample in block.keys():
        if samples_to_pop_map.get(sample) == population:
            i = random.randint(0,1)
            haplotypes.append(block[sample][f"hap_{i+1}"])
    
    return haplotypes


def _subset_block_by_population(block, population, samples_to_pop_map):
    return [sample for sample in block if get_samples_to_pop_map.get(sample) == population]


def _sample_n_haplotypes_from_same_population(block, n):
    samples = random.sample([sample for sample in block.keys()], n)

    haplotypes = []
    for sample in samples:
        hap_idx = random.randint(0,1)
        haplotypes.append(block[sample][hap_idx])

    assert len(haplotypes) == n

    return haplotypes


def sample_two_haplotypes(blockdict, chr, block, populations, samples_to_pop_map):
    block = blockdict[chr][block]
    
    if not isinstance(populations, list):
        populations = list(populations)

    if len(populations) == 1:
        pop_block = _subset_block_by_population(block, populations[0], samples_to_pop_map)
        haplotypes = _sample_n_haplotypes_from_same_population(pop_block, 2)
    elif len(populations) == 2:
        pop1_block = _subset_block_by_population(block, populations[0], samples_to_pop_map)
        pop2_block = _subset_block_by_population(block, populations[1], samples_to_pop_map)
        haplotypes = [_sample_n_haplotypes_from_same_population(pop_block, 1) for pop_block in [pop1_block, pop2_block]]
    else:
        raise ValueError("length of 'populations' argument must be either 1 or 2")

    return haplotypes
        

def get_samples_to_pop_map(samples_txt):
    samples = pd.read_csv(samples_txt, header=None)
    samples.columns = ["sample", "population"]
    samples_dict = dict(zip(samples["sample"], samples["population"]))
    assert len(list(set(samples_dict.values()))) == 2, f"{len(list(set(samples_dict.values())))} populations detected in {samples_txt} but only 2 are allowed"
    return samples_dict
    
def s_between_two_arrs(arr1, arr2):
    assert len(arr1) == len(arr2), "Haplotype blocks must be of same length"    
    return len(arr1) - np.count_nonzero(arr1 == arr2)

# def s_count_block(blockdict, chr, block, samples_to_pop_map):
    
#     population_names = list(set(samples_to_pop_map.values()))
    
#     # pop1_haps = block_haplotypes_per_population(blockdict, chr, block, population=population_names[0], samples_to_pop_map=samples_to_pop_map)
#     # pop2_haps = block_haplotypes_per_population(blockdict, chr, block, population=population_names[1], samples_to_pop_map=samples_to_pop_map)

#     # x1 = counts_to_dict([s_between_two_arrs(a,b) for a,b in itertools.combinations(pop1_haps, 2)])
#     # x2 = counts_to_dict([s_between_two_arrs(a,b) for a,b in itertools.combinations(pop2_haps, 2)])
#     # x3 = counts_to_dict([s_between_two_arrs(a,b) for a,b in itertools.product(pop1_haps, pop2_haps)])

#     return x1, x2, x3

def counts_to_dict(x):
    """
    Generate dictionary of counts of nucleotide difference (s-value) occurrences.
    :param list x: list of nt differences between loci
    :return: s_dict
    """
    s, s_count = np.unique(x, return_counts=True)
    return dict(zip(s,s_count))
    
def s_count(blockdict, samples_to_pop_map):

    x1 = []
    x2 = []
    x3 = []

    populations = list(set(samples_to_pop_map.values()))

    for chr in blockdict.keys():
        for block in blockdict[chr].keys():
            samples = random.sample([sample for sample in blockdict[chr][block].keys()], 2)
            haplotypes = random.choices(["hap_1", "hap_2"], k=2) # with replacement
            sampled_haps = [blockdict[chr][block][samples[i]][haplotypes[i]] for i in [0,1]]
            s = s_between_two_arrs(sampled_haps[0], sampled_haps[1])

            pop1 = samples_to_pop_map.get(samples[0])
            pop2 = samples_to_pop_map.get(samples[1])
            assert pop1 in populations, print(f"{pop1} not in populations {populations}")
            assert pop2 in populations, print(f"{pop2} not in populations {populations}")

            if pop1 != pop2:
                x3.append(s)
            elif pop1 == pop2 == populations[0]:
                x1.append(s)
            else: # pop1 == pop2 == populations[1]
                x2.append(s)

    s_dicts = [counts_to_dict(x) for x in [x1, x2, x3]]

    return s_dicts

def vcf_to_s_count(vcf_path, samples_path, block_length=64, n_blocks=10000):
    bd = BlocksDictionary()
    blockdict = bd.create(vcf_path, block_length=block_length, n_blocks=n_blocks, make_blocks=True)
    samples_map = get_samples_to_pop_map(samples_txt=samples_path)
    return s_count(blockdict=blockdict, samples_to_pop_map=samples_map)

def _sum_list_of_dicts_by_key(dicts):
    return dict(functools.reduce(operator.add, map(collections.Counter, dicts)))

def block_vcfs_to_s_count(directory, samples_path, file_pattern="*.vcf"):
    """Read in and process multiple VCFs from independent simulations, each containing a block"""
    samples_map = get_samples_to_pop_map(samples_txt=samples_path)

    x1 = []
    x2 = []
    x3 = []
    populations = list(set(samples_map.values()))

    for vcf_path in tqdm.tqdm(glob.glob(f"{directory}/{file_pattern}")):
        bd = BlocksDictionary()
        blockdict = bd.create(vcf_path, make_blocks=False)

        samples = random.sample([sample for sample in blockdict["1"]["1"].keys()], 2)
        haplotypes = random.choices(["hap_1", "hap_2"], k=2) # with replacement
        sampled_haps = [blockdict["1"]["1"][samples[i]][haplotypes[i]] for i in [0,1]]
        s = s_between_two_arrs(sampled_haps[0], sampled_haps[1])

        pop1 = samples_map.get(samples[0])
        pop2 = samples_map.get(samples[1])
        assert pop1 in populations, print(f"{pop1} not in populations {populations}")
        assert pop2 in populations, print(f"{pop2} not in populations {populations}")

        if pop1 != pop2:
                x3.append(s)
        elif pop1 == pop2 == populations[0]:
                x1.append(s)
        else: # pop1 == pop2 == populations[1]
                x2.append(s)

    s_dicts = [counts_to_dict(x) for x in [x1, x2, x3]]
    return s_dicts

