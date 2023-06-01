# this file will eventually replace preprocess.py

import allel
import zarr
import numpy as np
import math
from collections import defaultdict
import itertools


class NestedDefaultDict(defaultdict):
    """From https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict"""

    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(
            NestedDefaultDict, *args, **kwargs)

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
        allel.vcf_to_zarr(vcf_path, self.zarr_path,
                          fields=["samples", "calldata/GT",
                                  "variants/CHROM", "variants/POS"],
                          overwrite=True)


class BlocksDictionary:

    def __init__(self, callset, block_len, gap, max_missing, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callset = callset
        self.block_len = block_len
        self.gap = gap
        self.max_missing = max_missing
        self.blockdict = self.create()
        self.n_blocks = self.num_blocks()

    def __repr__(self):
        return str(dict(self.blockdict))

    def __getitem__(self, item):
        return self.blockdict[item]

    def keys(self):
        return self.blockdict.keys()

    def values(self):
        return self.blockdict.values()

    def samples(self):
        samples = [[list(self.blockdict[chrom][block].keys())
                    for block in self.blockdict[chrom].keys()] for chrom in self.blockdict.keys()]
        return set(list(itertools.chain(*itertools.chain(*samples))))

    def num_blocks(self):
        return sum([len(self.blockdict[chr].keys()) for _, chr in enumerate(self.blockdict.keys())])

    @staticmethod
    def n_missing(arr):
        return np.sum(arr < 0)

    def block_indices_chr(self, chr_name):
        chrom_indices = np.where(self.callset.chrom == chr_name)
        min_pos = min(self.callset.pos[chrom_indices])
        max_pos = max(self.callset.pos[chrom_indices])
        chrom_len = max_pos-min_pos

        n_blocks = math.floor(chrom_len/(self.gap+self.block_len))
        block_start_pos = np.array(
            [i*(self.gap+self.block_len) for i in range(n_blocks)]) + min_pos

        block_end_positions = block_start_pos + self.block_len
        blocks_idx = list(zip(block_start_pos, block_end_positions))

        return blocks_idx

    def create(self):
        blocks_dictionary = NestedDefaultDict()

        for chrom in list(set(self.callset.chrom)):
            block_idx = self.block_indices_chr(chrom)
            for (start_idx, end_idx) in block_idx:
                for sample_idx, sample_name in enumerate(self.callset.samples):
                    hap1 = self.callset.gt[start_idx:end_idx,
                                           sample_idx][:, 0]
                    hap2 = self.callset.gt[start_idx:end_idx,
                                           sample_idx][:, 1]
                    if (self.n_missing(hap1) <= self.max_missing) and (self.n_missing(hap2) <= self.max_missing):
                        blocks_dictionary[chrom][f"{chrom}:{start_idx}-{end_idx}"][sample_name][f"hap_1"] = hap1
                        blocks_dictionary[chrom][f"{chrom}:{start_idx}-{end_idx}"][sample_name][f"hap_2"] = hap2

        return blocks_dictionary
