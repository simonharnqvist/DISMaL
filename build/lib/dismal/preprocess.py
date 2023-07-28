import random
import allel
import pandas as pd
import numpy as np
import zarr
import os
import itertools


# def _trim_blocks(df, blocklen):
#    """Superceded by _expand_intron_to_blocks"""
#     # need to split introns
#     start_indices = [random.randint(
#         df.iloc[row, 3], (df.iloc[row, 4]-blocklen)) for row in range(0, len(df))]
#     end_indices = np.array(start_indices) + blocklen+1
#     return pd.DataFrame({"chr": df.iloc[:, 0], "start": start_indices, "end": end_indices})


def _filter_ld(df, ld_bp):
    sorted = df.sort_values(["chr", "start"])

    filtered_dfs = []
    for chromosome in df["chr"].unique():
        df_chr = sorted[sorted["chr"] == chromosome]
        filtered = np.where([sorted.iloc[row, 1] - df_chr.iloc[row-1, 2]
                            > ld_bp for row in range(1, len(df_chr)-1)])
        filtered_dfs.append(df_chr.iloc[filtered[0], :])

    return pd.concat(filtered_dfs)


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


def gff3_to_blocks(gff3_path, blocklen, ld_dist_bp):
    """Subset GFF3 file for intronic blocks of length blocklen, at least ld_dist apart"""

    pd.set_option('chained_assignment', None)

    print(gff3_path)
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

    # Filter for LD
    blocks_ld_filtered = _filter_ld(blocks_unfiltered, ld_dist_bp)

    # Include block information in output df
    blocks_ld_filtered.loc[:, "block_id"] = blocks_ld_filtered.loc[:, "chr"] + ":" + \
        blocks_ld_filtered.loc[:, "start"].astype(
            str) + "-" + blocks_ld_filtered.loc[:, "end"].astype(str)

    return blocks_ld_filtered


class CallSet:
    """Class to represent a skallele callset (i.e. a VCF)"""

    def __init__(self, vcf_path=None, zarr_path=None):

        if zarr_path is None:
            self.zarr_path = "zarrstore"
        else:
            self.zarr_path = zarr_path

        if vcf_path is None:
            assert zarr_path is not None

        self.vcf_path = vcf_path
        if not os.path.exists(self.zarr_path):
            allel.vcf_to_zarr(self.vcf_path, self.zarr_path,
                              fields=["samples", "calldata/GT", "variants/CHROM", "variants/POS"], overwrite=False)

        self.callset = zarr.open_group(zarr_path, mode='r')
        self.chromosomes = self.callset["variants/CHROM"][:]
        self.samples = self.callset["samples"][:]
        self.gt = self.callset["calldata/GT"][:]
        self.pos = self.callset["variants/POS"][:]
        self.callset_positions_df = pd.DataFrame(
            {"chr": self.chromosomes, "pos": self.pos})

        self.callset_positions_df["gt_idx"] = self.callset_positions_df.index


def block_snps(callset, blocks):
    """Merge block and callset information to get dataframe of SNPs and which block they belong to"""
    return blocks.merge(callset.callset_positions_df, how='left', on="chr") \
        .query('pos.between(`start`, `end`)')


def block_callset(callset, gt_idxs, ploidy=2):
    """Get the callset for a block defined by GT indices, divided by haplotype"""
    block_calls = pd.DataFrame(list(itertools.product(callset.samples, list(range(ploidy)))),
                               columns=["sample", "hap"])

    for idx in gt_idxs:
        block_calls[idx] = callset.gt[idx].flatten()

    block_calls = block_calls.dropna()
    return block_calls


def n_segr_sites(block_callset):
    """Count segregating sites between samples of length 2 from a block callset"""
    return sum(block_callset.iloc[0, 2:] != block_callset.iloc[1, 2:])


def get_segregating_sites_spectrum(callset, block_snps, samples_map_df, blocklen, ploidy=2, sampling_probabilities=[0.25, 0.25, 0.5]):

    rng = np.random.default_rng()

    populations = samples_map_df["population"].unique()
    pop1 = list(samples_map_df["sample"]
                [samples_map_df["population"] == populations[0]])
    pop2 = list(samples_map_df["sample"]
                [samples_map_df["population"] == populations[1]])

    sss = np.zeros(shape=(3, blocklen))

    for block in block_snps["block_id"].unique():
        block_gt_idxs = list(
            block_snps[block_snps["block_id"] == block]["gt_idx"])
        block_callset_df = block_callset(callset,
                                         block_gt_idxs,
                                         ploidy=ploidy)

        sampling_state = int(np.where(rng.multinomial(
            n=1, pvals=sampling_probabilities) == 1)[0]) + 1

        if sampling_state == 1:
            block_sss = n_segr_sites(
                block_callset_df[block_callset_df["sample"].isin(pop1)].sample(n=2))
        elif sampling_state == 2:
            block_sss = n_segr_sites(
                block_callset_df[block_callset_df["sample"].isin(pop2)].sample(n=2))
        else:
            assert sampling_state == 3
            block_sss = n_segr_sites(pd.concat([block_callset_df[block_callset_df["sample"].isin(pop1)].sample(n=1),
                                     block_callset_df[block_callset_df["sample"].isin(pop2)].sample(n=2)]))

        sss[sampling_state-1, int(block_sss)
            ] = sss[sampling_state-1, int(block_sss)] + 1

    return sss


class SegregatingSitesSpectrum:

    def __init__(self,
                 gff3_path,
                 vcf_path,
                 zarr_path,
                 samples_map,
                 blocklen=100,
                 ld_dist_bp=1000,
                 ploidy=2,
                 sampling_probabilities=[0.25, 0.25, 0.5]):

        self.gff3_path = gff3_path,
        self.vcf_path = vcf_path,
        self.zarr_path = zarr_path,
        self.samples_map = pd.read_csv(samples_map)
        self.blocklen = blocklen,
        self.ld_dist_bp = ld_dist_bp,
        self.ploidy = ploidy,
        self.sampling_probabilities = sampling_probabilities

        self.blocks = gff3_to_blocks(
            self.gff3_path,
            blocklen=self.blocklen,
            ld_dist_bp=self.ld_dist_bp)

        self.callset = CallSet(self.vcf_path, self.zarr_path)

        self.block_snps = block_snps(self.callset, self.blocks)

        self.s3 = get_segregating_sites_spectrum(self.callset,
                                                 block_snps=self.block_snps_df,
                                                 samples_map_df=self.samples_map,
                                                 blocklen=blocklen,
                                                 ploidy=ploidy,
                                                 sampling_probabilities=sampling_probabilities)

    def __repr__(self):
        return self.s3

    def __str__(self):
        return self.s3

    def calculate(self):
        rng = np.random.default_rng()

        populations = self.samples_map["population"].unique()
        pop1 = list(self.samples_map["sample"]
                    [self.samples_map["population"] == populations[0]])
        pop2 = list(self.samples_map["sample"]
                    [self.samples_map["population"] == populations[1]])

        sss = np.zeros(shape=(3, self.blocklen))

        for block in self.block_snps["block_id"].unique():
            block_gt_idxs = list(
                self.block_snps[block_snps["block_id"] == block]["gt_idx"])
            block_callset_df = block_callset(self.callset,
                                             block_gt_idxs,
                                             ploidy=self.ploidy)

        sampling_state = int(np.where(rng.multinomial(
            n=1, pvals=self.sampling_probabilities) == 1)[0]) + 1

        if sampling_state == 1:
            block_sss = n_segr_sites(
                block_callset_df[block_callset_df["sample"].isin(pop1)].sample(n=2))
        elif sampling_state == 2:
            block_sss = n_segr_sites(
                block_callset_df[block_callset_df["sample"].isin(pop2)].sample(n=2))
        else:
            assert sampling_state == 3
            block_sss = n_segr_sites(pd.concat([block_callset_df[block_callset_df["sample"].isin(pop1)].sample(n=1),
                                     block_callset_df[block_callset_df["sample"].isin(pop2)].sample(n=2)]))

        sss[sampling_state-1, int(block_sss)
            ] = sss[sampling_state-1, int(block_sss)] + 1

        return sss
