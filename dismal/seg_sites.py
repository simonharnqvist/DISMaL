from dismal.blockgt import BlockGT
from dismal.blocks import gff3_to_blocks
from dismal.callset import CallSet
from collections import Counter
import numpy as np
import tqdm
import pandas as pd
import seaborn as sns
import warnings

class SegregatingSitesSpectrum:

    def __init__(self, blocks_df, callset, samples_map, s1_deme=None, s2_deme=None):
        self.chromosomes = list(blocks_df["chr"])
        self.block_starts = list(blocks_df["start"])
        self.block_ends = list(blocks_df["end"])
        self.block_lengths = list(blocks_df["end"]-blocks_df["start"])

        if s1_deme is None:
            self.pop1 = samples_map.iloc[:, 1].unique()[0]
        else:
            assert s1_deme in list(samples_map.iloc[:, 1].unique())
            self.pop1 = s1_deme

        if s2_deme is None:
            self.pop2 = samples_map.iloc[:, 1].unique()[1]
        else:
            assert s2_deme in list(samples_map.iloc[:, 1].unique())
            self.pop2 = s2_deme

        self.pop1_samples = samples_map.iloc[:,0][samples_map.iloc[:, 1] == self.pop1]
        self.pop2_samples = samples_map.iloc[:,0][samples_map.iloc[:, 1] == self.pop2]
        
        block_gt_arrs = [callset.gt[np.where((callset.chromosomes == chromosome) 
                                             & (callset.pos >= start) 
                                             & (callset.pos <= end))] for (chromosome, start, end) 
            in tqdm.tqdm(list(zip(self.chromosomes, self.block_starts, self.block_ends)))]
        
        self.block_gts = [
            BlockGT(arr, 
                    callset_samples=callset.samples, 
                    pop1_samples=self.pop1_samples, 
                    pop2_samples=self.pop2_samples) for arr in block_gt_arrs
        ]
        
        self.s1, self.s2, self.s3 = self.seg_sites_distr()

        self.visualisation = self.visualise_sdistr(self.s1, self.s2, self.s3)


    def seg_sites_distr(self):
        """Compute segregating sites spectrum from blocks"""

        seg_sites_spec = []

        s1_counter = Counter([item for sublist in [block.s1 for block in self.block_gts] for item in sublist]) 
        s2_counter = Counter([item for sublist in [block.s2 for block in self.block_gts] for item in sublist])
        s3_counter = Counter([item for sublist in [block.s3 for block in self.block_gts] for item in sublist])

        for s_counter in [s1_counter, s2_counter, s3_counter]:
            s_max = max(s_counter.keys()) + 1
            s_arr = np.zeros(s_max)
            s_arr[list(s_counter.keys())] = list(s_counter.values())
            seg_sites_spec.append(s_arr)

        return seg_sites_spec
    
    
    @staticmethod
    def from_vcf_gff3(gff3_path,
                      samples_map_path, 
                      blocklen,
                      min_block_distance,
                      vcf_path=None, 
                      vcf_npz_path=None, 
                      blocks_parquet_path=None, 
                      out_npz_path=None,
                      select_chromosomes=None,
                      exclude_chromosomes=None,
                      genomic_partition="intron",
                      trim_starts=10,
                      trim_ends=10):
        """Create SegregatingSitesSpectrum from VCF and GFF3 files.

        Args:
            vcf_path (path or str): Path to VCF.
            gff3_path (path or str): Path to GFF3.
            samples_map_path (path or str): Path to samples map, a CSV file with columns "sample" and "population"
            blocklen (int): Length of genomic blocks to use in segregating sites calculation.
            min_block_distance (int): Minimum distance (in bp) between two blocks.
            vcf_npz_path (path, optional): Path to Npz array containing processed VCF, if this step is already done. Defaults to None.
            blocks_parquet_path (path, optional): Path to parquet store of blocks dataframe. Defaults to None.
            out_npz_path (path, optional): Path to which to write Npz array of segregating sites spectrum. Defaults to None.
            genomic_partition (str, optional): Which genomic element to use to make blocks. Defaults to "intron".

        Returns: SegregatingSitesSpectrum object
        """

        if vcf_path is None:
            assert vcf_npz_path is not None, "Please provide either VCF or NPZ"

        print("Making blocks from GFF3...")
        blocks_df = gff3_to_blocks(gff3_path=gff3_path, 
                                   blocklen=blocklen, 
                                   min_block_distance=min_block_distance, 
                                   genomic_partition=genomic_partition, 
                                   select_chromosomes=select_chromosomes, 
                                   exclude_chromosomes=exclude_chromosomes,
                                   trim_starts=trim_starts, trim_ends=trim_ends,
                                   parquet_path=blocks_parquet_path)
        
        print(f"Generated {len(blocks_df)} blocks")
        
        print("Reading VCF to CallSet...")
        callset = CallSet(vcf_path=vcf_path, npz_path=vcf_npz_path)

        print("Creating SegregatingSitesSpectrum...")
        samples_map = pd.read_csv(samples_map_path)
        seg_sites_spec = SegregatingSitesSpectrum(blocks_df, callset, samples_map=samples_map)

        if out_npz_path is not None:
            np.savez(out_npz_path, 
                     s1=seg_sites_spec.s1, 
                     s2=seg_sites_spec.s2, 
                     s3=seg_sites_spec.s3)
            
        return seg_sites_spec
    
    @staticmethod
    def visualise_sdistr(s1, s2, s3):
        """Visualise s distributions"""
        sdistr_df = pd.concat([pd.DataFrame(list(zip([sdist_name]*len(sdist), 
                                          [i for i in range(len(sdist))], 
                                          sdist)), columns=["distr", "s", "count"]) 
                                          for (sdist, sdist_name) in list(zip([s1, s2, s3], ["s1", "s2", "s3"]))])
    
        warnings.simplefilter(action='ignore', category=FutureWarning)
        g = sns.FacetGrid(data=sdistr_df, col="distr", sharex=False, sharey=False)
        g.map(sns.barplot, "s", "count")
        g.fig.text(x=0.15, y=1, s="s1 (within species 1)", fontsize=10, fontstyle="italic")
        g.fig.text(x=0.45, y=1, s="s2 (within species 2)", fontsize=10, fontstyle="italic")
        g.fig.text(x=0.75, y=1, s="s3 (between species)", fontsize=10)
        g.set_xticklabels(rotation=90)
        max_s = len(s3)
        g.set(xticks=np.arange(0, max_s, 5))

        return g