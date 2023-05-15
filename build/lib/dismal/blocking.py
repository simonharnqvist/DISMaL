# this file will eventually replace preprocess.py

import allel
import zarr
import numpy as np
import math
from collections import defaultdict

class NestedDefaultDict(defaultdict):
    """From https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict"""
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return str(dict(self))

class CallSet:

    def __init__(self, zarr_path):
        self.zarr_path = zarr_path
        self.callset = zarr.open_group(self.zarr_path, mode='r')
        self.chrom = self.callset["variants/CHROM"][:]
        self.samples = self.callset["samples"][:]
        self.gt = self.callset["calldata/GT"][:]
        self.pos = self.callset["variants/POS"][:]

    def generate_from_vcf(self, vcf_path):
        allel.vcf_to_zarr(self.vcf_path, self.zarr_path,
                           fields=["samples", "calldata/GT", "variants/CHROM", "variants/POS"], overwrite=True)
        

class BlocksDictionary:

    def __init__(self, callset, block_len, gap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callset = callset
        self.block_len = block_len
        self.gap = gap
        self.blockdict = self.create()

    def __repr__(self):
        return str(dict(self.blockdict))
    
    def __getitem__(self, item):
        return self.blockdict[item]
    
    def keys(self):
        return self.blockdict.keys()
    
    def values(self):
        return self.blockdict.values()

    def block_indices_chr(self, chr_name):
        chr_indices = np.where(self.callset.chrom == chr_name)
        min_pos = min(self.callset.pos[chr_indices])
        max_pos = max(self.callset.pos[chr_indices])
        chr_len = max_pos-min_pos

        n_blocks = math.floor(chr_len/(self.gap+self.block_len))
        block_start_pos = np.array([i*(self.gap+self.block_len) for i in range(0, n_blocks)]) + min_pos

        block_end_positions = block_start_pos + self.block_len
        blocks_idx = list(zip(block_start_pos, block_end_positions))
    
        return blocks_idx

    def create(self):
        blocks_dictionary = NestedDefaultDict()

        for chr in list(set(self.callset.chrom)):
            block_idx = self.block_indices_chr(chr)
            for (start_idx, end_idx) in block_idx:
                for i in range(0, len(self.callset.samples)):
                    for hap in [0,1]:
                        blocks_dictionary[chr][f"{chr}:{start_idx}-{end_idx}"][self.callset.samples[i]][f"hap_{hap+1}"] = self.callset.gt[start_idx:end_idx, i][:,hap]

        return blocks_dictionary
    