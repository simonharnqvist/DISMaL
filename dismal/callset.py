import allel
import pandas as pd
import numpy as np
import os

class CallSet:
    """Class to represent a skallele callset (i.e. a VCF)"""

    def __init__(self, vcf_path=None, npz_path=None):

        if vcf_path is None:
            assert npz_path is not None

        self.vcf_path = vcf_path
        self.npz_path = npz_path
        if self.vcf_path is None and not os.path.exists(self.npz_path):
            self.npz_path = "vcf.npz"
            allel.vcf_to_npz(self.vcf_path, self.npz_path,
                              fields=["samples", "calldata/GT", "variants/CHROM", "variants/POS"], overwrite=False)

        self.callset = np.load(self.npz_path, allow_pickle=True)
        self.chromosomes = self.callset["variants/CHROM"][:]
        self.chromosomes_int = self.integer_encode_chromosomes()
        self.samples = self.callset["samples"][:]
        self.gt = self.callset["calldata/GT"][:]
        self.pos = self.callset["variants/POS"][:]
        self.callset_positions_df = pd.DataFrame(
            {"chr": self.chromosomes, "pos": self.pos})

        self.callset_positions_df["gt_idx"] = self.callset_positions_df.index


    def integer_encode_chromosomes(self):
        """Convert array ["chr1", "chr1", "chr1", "chr2", ..., "chrN"] 
        to [0,0,0,1,...(N-1)]"""
        
        chromosomes = self.chromosomes
    
        uniq_chromosome_names, chr_idxs = np.unique(chromosomes, return_index=True)
        uniq_chromosome_names = uniq_chromosome_names[np.argsort(chr_idxs)]
    
        for uniq_chrom_idx, uniq_chrom in np.ndenumerate(uniq_chromosome_names):
            chromosomes[np.where(chromosomes == uniq_chrom)] = uniq_chrom_idx

        return chromosomes.astype(int)