import allel
import pandas as pd
import numpy as np
import os
import itertools
import random
from collections import Counter, deque
import tqdm
from numba import jit
import time

def _filter_ld(blocks_df, min_block_distance):
    """Filter blocks for minimum distance between them to reduce effect of LD"""
    filtered_rows = [dict(blocks_df.iloc[0, :])]

    for index, row in blocks_df.iterrows():
        if (blocks_df.loc[index, "chr"] != filtered_rows[-1]["chr"]): # if different chromosomes, add block
            filtered_rows.append(dict(blocks_df.iloc[index, :]))
        elif (blocks_df.loc[index, "start"] - filtered_rows[-1]["end"] > min_block_distance): # if same chromosome, check distance, add if far enough apart
            filtered_rows.append(dict(blocks_df.iloc[index, :]))
        else:
            pass

    return pd.DataFrame(filtered_rows)


def _expand_intron_to_blocks(row, blocklen):
    """Expand an intron (row) to blocks of length blocklen"""
    start = row["start"]
    end = row["end"]
    assert end - start >= blocklen

    start_idxs = list(range(start, end, blocklen+1))
    end_idxs = [i+blocklen for i in start_idxs]

    if end_idxs[-1] > end:  # if last block overshoots
        start_idxs.pop(-1)
        end_idxs.pop(-1)

    intron_blocks = pd.DataFrame(columns=["chr", "start", "end"])
    intron_blocks["chr"] = [row["chr"]] * len(start_idxs)
    intron_blocks["start"] = start_idxs
    intron_blocks["end"] = end_idxs

    return intron_blocks


def gff3_to_blocks(gff3_path, blocklen, min_block_distance, parquet_path=None):
    """Subset GFF3 file for intronic blocks of length blocklen, at least ld_dist apart"""

    pd.set_option('chained_assignment', None)

    gff = allel.gff3_to_dataframe(gff3_path)[["seqid", "type", "start", "end"]]
    gff.columns = ["chr", "type", "start", "end"]

    # Subset intronic regions
    introns_df = gff[gff["type"] == "intron"]

    # Subset len(intron) > blocklen
    introns_min_blocklen = introns_df[(
        introns_df["end"] - introns_df["start"]) > blocklen]

    # Expand each intron to multiple blocks of length blocklen
    blocks_unfiltered = pd.concat([_expand_intron_to_blocks(introns_min_blocklen.iloc[row, :],
                                                            blocklen=blocklen) for row in range(len(introns_min_blocklen))])
    blocks_unfiltered = blocks_unfiltered.reset_index(drop=True)

    # Filter for LD
    blocks_ld_filtered = _filter_ld(blocks_unfiltered, min_block_distance)

    # Include block information in output df
    blocks_ld_filtered.loc[:, "block_id"] = blocks_ld_filtered.loc[:, "chr"] + ":" + blocks_ld_filtered.loc[:, "start"].astype(str) + "-" + blocks_ld_filtered.loc[:, "end"].astype(str)

    blocks = blocks_ld_filtered

    if parquet_path is not None:
        blocks.to_parquet(parquet_path)

    return blocks


class CallSet:
    """Class to represent a skallele callset (i.e. a VCF)"""

    def __init__(self, vcf_path=None, npz_path=None):

        if npz_path is None:
            self.npz_path = "vcf.npz"
        else:
            self.npz_path = npz_path

        if vcf_path is None:
            assert npz_path is not None

        self.vcf_path = vcf_path
        if not os.path.exists(self.npz_path):
            allel.vcf_to_npz(self.vcf_path, self.npz_path,
                              fields=["samples", "calldata/GT", "variants/CHROM", "variants/POS"], overwrite=False)

        self.callset = np.load(self.npz_path, allow_pickle=True)
        self.chromosomes = self.callset["variants/CHROM"][:]
        self.samples = self.callset["samples"][:]
        self.gt = self.callset["calldata/GT"][:]
        self.pos = self.callset["variants/POS"][:]
        self.callset_positions_df = pd.DataFrame(
            {"chr": self.chromosomes, "pos": self.pos})

        self.callset_positions_df["gt_idx"] = self.callset_positions_df.index

def block_genotype_array(callset, chrom, start, end):
    """Genotype array for block."""
    snps_idxs = np.where((callset.chromosomes == chrom) & (callset.pos >= start) & (callset.pos <= end))
    return callset.gt[snps_idxs]

def index_individuals_wo_missing_sites(block_gt_arr):
    """Return indices of individuals that have no missing sites in a block"""
    idxs_nonmissing = [np.where([(block_gt_arr[:, i] >= 0).all() 
                                 for i in range(len(block_gt_arr[0]))])]
    return idxs_nonmissing[0][0]

def n_segr_sites(arr):
    """Count number of segregating sites between two sampled haplotype blocks."""
    assert (arr >= 0).all(), "Missing sites detected; please remove them before calculating segregating sites."
    return np.sum(arr[:, 0] != arr[:, 1])


def deques_to_s_matrix(s_deques):
    """Convert a list of 3 deques to a matrix of segregating sites."""
    array_lengths = [int(max(Counter(s_deques[i]).keys())) + 1 for i in [0,1,2]]
    max_s_value = max(array_lengths)
    return np.array([[Counter(s_deques[i])[s] for s in range(max_s_value)] for i in [0, 1, 2]])

def blockwise_segregating_sites(blocks, callset, samples, sampling_probs=[0.25, 0.25, 0.50]):
    """Calculate the number of segregating sites between sampled haplotypes for each block."""

    pops_names = samples["population"].unique()
    assert len(pops_names) == 2
    pops_idxs = [np.where(np.array(samples.set_index("sample").loc[callset.samples, :]["population"]) == pop_name) for pop_name in pops_names]

    rng = np.random.default_rng()

    S = [list(), list(), list()] # state 1,2,3
    
    for block_idx in tqdm.tqdm(range(len(blocks))):

        block_state = np.where(rng.multinomial(n=1, pvals=sampling_probs))[0][0] + 1

        block_start, block_end, block_chr = (blocks.loc[block_idx, "start"],
                                              blocks.loc[block_idx, "end"],
                                                blocks.loc[block_idx, "chr"])
        block_gt_arr = block_genotype_array(callset, block_chr, block_start, block_end)
        if len(block_gt_arr) == 0:
            continue

        indivs_wo_missing_sites = index_individuals_wo_missing_sites(block_gt_arr)
        if index_individuals_wo_missing_sites == 0:
            continue

        pop1_nomissing = np.intersect1d(pops_idxs[0], indivs_wo_missing_sites)
        pop2_nomissing = np.intersect1d(pops_idxs[1], indivs_wo_missing_sites)

        if block_state == 1:
            if len(pop1_nomissing) < 2:
                continue
            else:
                samples_idxs = rng.choice(pop1_nomissing, 2)
        elif block_state == 2:
            if len(pop2_nomissing) < 2:
                continue
            else:
                samples_idxs = rng.choice(pop2_nomissing, 2)
        else:
            assert block_state == 3
            if len(pop1_nomissing) == 0 or len(pop2_nomissing) == 0:
                continue
            else:
                samples_idxs = np.array([rng.choice(pop1_nomissing, 1), rng.choice(pop2_nomissing, 1)])

        ploidy = len(block_gt_arr[0][0])
        haplotype_idx = random.choice(range(0, ploidy))
        sample_genotypes = block_gt_arr[:, samples_idxs, haplotype_idx]

        s = n_segr_sites(sample_genotypes)

        S[block_state-1].append(s)


    return S

def from_vcf_gff3(vcf, gff3, samples_map, blocklen=100, min_block_distance=10_000,
                   sampling_probs=[0.25, 0.25, 0.50],
                     vcf_npz_path=None, block_parquet_path=None, segregating_sites_npy_path=None):
    """Generate segregating sites spectrum from a VCF and GFF3 annotation. 

    Args:
        vcf (str): Path to VCF.
        gff3 (str): Path to GFF3 annotation.
        samples_map (str): Path to comma-separated file that maps sample ID -> population.
        blocklen (int, optional): Length of genomic blocks to make. Defaults to 100.
        min_block_distance (int, optional): Minimum distance in base pairs between blocks. Defaults to 10_000.
        sampling_probs (list, optional): The weighting of each sampling state. Defaults to [0.25, 0.25, 0.50].
        vcf_npz_path (str, optional): Path to which the npz of the VCF is written/read from. Defaults to None.
        block_parquet_path (str, optional): Path to which the dataframe of blocks is written/read from. Defaults to None.
        segregating_sites_npy_path (str, optional): Path to which the matrix of segregating sites is written. Defaults to None.

    Returns:
        segregating_sites_spectrum: np.array 3 x s of the count of s segregating sites in each state.
    """

    start_time = time.time()
    
    if block_parquet_path is not None and os.path.exists(block_parquet_path):
        print(f"{time.time() - start_time}s: Reading blocks from parquet at {block_parquet_path}...")
    else:
        print("Processing GFF3 annotation to blocks...")
    blocks = gff3_to_blocks(gff3_path=gff3, blocklen=blocklen, min_block_distance=min_block_distance, parquet_path=block_parquet_path)

    if vcf_npz_path is None:
        print(f"{time.time() - start_time}s: Processing VCF file to npz storage at {vcf_npz_path}...")
    else:
        print(f"{time.time() - start_time}s: Reading callset information from NumPy storage at {vcf_npz_path}...")
    callset = CallSet(vcf_path=vcf, npz_path=vcf_npz_path)

    print(f"{time.time() - start_time}s: Reading in samples map {samples_map}...")
    samples = pd.read_csv(samples_map)

    print("{time.time() - start_time}s: Calculating segregating sites spectrum...")
    segregating_sites_spectrum = blockwise_segregating_sites(blocks=blocks, callset=callset, samples=samples, sampling_probs=sampling_probs)

    print(f"{time.time() - start_time}s: Saving segregating sites spectrum to {segregating_sites_npy_path}")
    if segregating_sites_npy_path is not None:
        np.save(file=segregating_sites_npy_path, arr=segregating_sites_spectrum)



    return segregating_sites_spectrum