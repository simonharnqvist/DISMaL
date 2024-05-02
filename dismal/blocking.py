from pyranges import PyRanges
import pyranges as pr
import itertools
import pandas as pd
import tqdm
import numpy as np
import csv
from collections import Counter
import allel
from dismal.callset import CallSet

np.set_printoptions(suppress=True)


def _sample_map_to_sample_idxs(sample_map, callset):
    """Get VCF indices per population."""
    if type(sample_map) is str:
        with open(sample_map, mode='r') as infile:
            reader = csv.reader(infile)
            sample_map = {row[0]: row[1] for row in reader}

    assert type(sample_map) is dict
    pops = np.unique(list(sample_map.values()))
    pop1_idxs = np.where([sample_map.get(sample) == pops[0]
                         for sample in callset.samples])[0]
    pop2_idxs = np.where([sample_map.get(sample) == pops[1]
                         for sample in callset.samples])[0]

    return pop1_idxs, pop2_idxs


def _estimate_dxy(callset, sample_map):
    """Estimate dxy from first chromosome in CallSet"""

    pop1_idxs, pop2_idxs = _sample_map_to_sample_idxs(sample_map, callset)
    chrom = np.unique(callset.chromosomes)[0]
    calls = callset.gt[callset.chromosomes == chrom]
    pos = callset.pos[callset.chromosomes == chrom]
    pop1_ac = allel.GenotypeArray(calls[:, pop1_idxs, :]).count_alleles()
    pop2_ac = allel.GenotypeArray(calls[:, pop2_idxs, :]).count_alleles()

    return allel.sequence_divergence(pos, pop1_ac, pop2_ac)


def blocklen_using_dxy(callset: CallSet, sample_map: dict, numerator: int = 3) -> float:
    """Rule-of-thumb blocklength estimation using 3/dxy rule.

    :param callset: Call data.
    :type callset: CallSet
    :param sample_map: Sample:population map.
    :type sample_map: dict
    :param numerator: Desired mean of between-distribution, defaults to 3
    :type numerator: int, optional
    :return: Recommended blocklength.
    :rtype: float
    """
    dxy = _estimate_dxy(callset, sample_map)
    return round(numerator/dxy)


def _get_callable_sites_per_sample(callable_sites):
    """Make PyRanges/BED of callable_sites regions for each sample from mosdepth type PyRanges."""
    callable_sites_df = callable_sites.df.drop("Name", axis=1)
    callable_sites_df["Score"] = callable_sites_df["Score"].str.split(",")
    callable_sites_df = callable_sites_df.explode(
        "Score").rename({"Score": "Sample"})
    return PyRanges(callable_sites_df)

# TODO: intersect with intergenic space instead


def _subset_features(annotation, features):
    """Subset annotation PyRanges for feature name e.g. 'intron'"""
    return PyRanges(annotation.df[annotation.df["Feature"].isin(features)])


def _trim_partitions(ranges, trim_start=0, trim_end=0):
    """UNTESTED: Trim each partition unit by trim_start at beginning and trim_end at end"""
    blocks_df_trimmed = ranges.df
    blocks_df_trimmed["start"] = blocks_df_trimmed["start"] + trim_start
    blocks_df_trimmed["end"] = blocks_df_trimmed["end"] - trim_end

    return PyRanges(blocks_df_trimmed)


def _subset_ranges_per_sample(ranges):
    """Subset PyRanges for each sample to generate list of PyRanges objects. Sample column must be 'Score'."""
    return [PyRanges(ranges.df[ranges.df["Score"] == sample]) for sample in set(ranges.df["Score"])]


def _intersect_samples(sample_ranges):
    """Intersect all samples with one another to find overlapping regions."""
    res = []
    for (sample1_idx, sample2_idx) in list(itertools.combinations(range(len(sample_ranges)), 2)):
        sample_intersect = sample_ranges[sample1_idx].intersect(
            sample_ranges[sample2_idx]).df
        merged = PyRanges(sample_intersect).merge(slack=1).df
        merged["sample1"] = sample_intersect["Score"]
        merged["sample2"] = sample_ranges[sample2_idx].df["Score"][0]
        res.append(merged)
    return PyRanges(pd.concat(res))


def _make_ranges(ranges, block_size):
    """Partition PyRanges object into blocks of size block_size."""
    windowed = ranges.window(window_size=block_size)
    windowed.Length = windowed.End - windowed.Start
    return windowed[windowed.Length >= block_size]


def _filter_ld(blocks, min_block_distance):
    """Filter blocks for minimum distance between them to reduce effect of LD"""
    filtered_rows = [dict(blocks.iloc[0, :])]

    print("Filtering on minimum block distance...")
    for index, row in tqdm.tqdm(blocks.iterrows()):
        # if different chromosomes, add block
        if (blocks.loc[index, "Chromosome"] != filtered_rows[-1]["Chromosome"]):
            filtered_rows.append(dict(blocks.iloc[index, :]))
        # if same chromosome, check distance, add if far enough apart
        elif (blocks.loc[index, "Start"] - filtered_rows[-1]["End"] > min_block_distance):
            filtered_rows.append(dict(blocks.iloc[index, :]))
        else:
            pass

    return pd.DataFrame(filtered_rows)


def make_random_blocks(callset: CallSet, block_size: int,
                       chrom_sizes: dict = None, blocks_per_pair: int = 1000) -> PyRanges:
    """Generate random blocks without considering annotation or coverage (e.g. if data is pre-filtered, or simulated).

    :param callset: Call data.
    :type callset: CallSet
    :param block_size: Size of genomic blocks.
    :type block_size: int
    :param chrom_sizes: Chromosome sizes, defaults to None in which case first and last positions are used.
    :type chrom_sizes: dict, optional
    :param blocks_per_pair: Number of blocks to make per pair of individuals, defaults to 1000
    :type blocks_per_pair: int, optional
    :return: Block coordinates
    :rtype: PyRanges
    """

    pairs = list(itertools.combinations(callset.samples, 2))
    n_blocks = blocks_per_pair * len(pairs)

    if chrom_sizes is None:
        chrom_sizes = pr.from_dict({
            "Chromosome": [chrom for chrom in set(callset.chromosomes)],
            "Start": [callset.pos[callset.chromosomes == chrom].min() for chrom in set(callset.chromosomes)],
            "End": [callset.pos[callset.chromosomes == chrom].max() for chrom in set(callset.chromosomes)]})

    rand_blocks = pr.random(n=n_blocks, length=block_size,
                            chromsizes=chrom_sizes, strand=False).df
    rand_blocks["sample1"] = [pair[0] for pair in pairs] * blocks_per_pair
    rand_blocks["sample2"] = [pair[1] for pair in pairs] * blocks_per_pair

    return PyRanges(rand_blocks)


def make_blocks(block_size: int, annotation: "PyRanges | str" = None,
                callable_sites: "PyRanges | str" = None,
                features: list[str] = None, trim_start: int = 10, trim_end: int = 10) -> PyRanges:
    """Make blocks with respect to an annotation and/or callable sites.

    :param block_size: Desired length of genomic blocks.
    :type block_size: int
    :param annotation: Genome annotation, defaults to None
    :type annotation: PyRanges | str, optional
    :param callable_sites: Callable sites, defaults to None
    :type callable_sites: PyRanges | str, optional
    :param features: Features to subset for. Will be removed in a future version., defaults to None
    :type features: list[str], optional
    :param trim_start: Length of sequence to trim, defaults to 10
    :type trim_start: int, optional
    :param trim_end: Length of sequence to trim, defaults to 10
    :type trim_end: int, optional
    :raises ValueError: If not at least one of callable_sites or annotation are provided.
    :return: Blocks
    :rtype: PyRanges
    """

    if callable_sites is not None:
        if isinstance(callable_sites, str):
            callable_sites = pr.read_bed(callable_sites)

        callable_sites_per_sample = _get_callable_sites_per_sample(
            callable_sites)

    if annotation is not None:
        if isinstance(annotation, str):
            annotation = pr.read_gff3(annotation)

        if features is None:
            features = ["intron"]

        subset_annotation = _subset_features(annotation, features)
        trimmed_subset_annotation = _trim_partitions(
            subset_annotation, trim_start=trim_start, trim_end=trim_end)

    if annotation is not None and callable_sites is not None:
        callable_sites_annotation = callable_sites_per_sample.intersect(
            trimmed_subset_annotation)
    elif annotation is not None:
        callable_sites_annotation = annotation
    elif callable_sites is not None:
        callable_sites_annotation = callable_sites_per_sample
    else:
        raise ValueError(
            "Either an annotation or a callability BED/ranges or both must be provided. Otherwise, use make_random_blocks()")

    sample_ranges = _subset_ranges_per_sample(callable_sites_annotation)
    intersected = _intersect_samples(sample_ranges)
    blocks = _make_ranges(intersected, block_size)

    return blocks


def _num_seg_sites(callset, chrom_idx, start, stop, sample1, sample2):
    """Number of segregating sites between two coordinates"""

    sample_idxs = np.where(callset.samples == sample1)[
        0][0], np.where(callset.samples == sample2)[0][0]
    hap_idx = np.random.default_rng().integers(0, 2)
    haps = callset.gt[(callset.chromosomes == chrom_idx) & (
        callset.pos >= start) & (callset.pos < stop)]
    hap1 = haps[:, sample_idxs[0], hap_idx]
    hap2 = haps[:, sample_idxs[1], hap_idx]

    arr = np.array([hap1, hap2])
    arr = arr[:, ~(arr == -1).any(axis=0)]

    return np.sum(arr[0] != arr[1])


def _blockwise_seg_sites(blocks, callset):

    return blocks.df.apply(lambda row: _num_seg_sites(callset, 0, row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4]),
                           axis=1)


def _determine_state(sample1, sample2, sample_map):
    """Assign Markov chain states to each sampled block."""

    if type(sample_map) is str:
        with open(sample_map, mode='r') as infile:
            reader = csv.reader(infile)
            sample_map = {row[0]: row[1] for row in reader}

    assert type(sample_map) is dict

    pops = list(set(sample_map.values()))
    pop1, pop2 = sample_map.get(sample1), sample_map.get(sample2)
    if pop1 != pop2:
        state = 3
    elif pop1 == pops[0] and pop1 == pop2:
        state = 1
    elif pop1 == pops[1] and pop1 == pop2:
        state = 2
    else:
        raise ValueError

    return state


def _get_s_distr(blocks_df, state):
    """Compute partial distribution of segregating sites from dataframe of blocks."""
    s_counter = Counter(blocks_df["NumSegSites"][blocks_df["state"] == state])
    s_max = max(s_counter.keys()) + 1
    s_arr = np.zeros(s_max)
    s_arr[list(s_counter.keys())] = list(s_counter.values())
    return s_arr


def segregating_sites_distribution(blocks: "PyRanges | str",
                                   sample_map: dict,
                                   save_blocks_bed: str = "blocks_with_state.bed",
                                   save_distr_npz: str = "s_distr.npz") -> tuple[np.array]:
    """Compute segregating sites distributions within and between populations.

    :param blocks: Blocks
    :type blocks: PyRanges | str
    :param sample_map: Map sample:population
    :type sample_map: dict
    :param save_blocks_bed: Path to save blocks to, defaults to "blocks_with_state.bed"
    :type save_blocks_bed: str, optional
    :param save_distr_npz: Path to save distributions to, defaults to "s_distr.npz"
    :type save_distr_npz: str, optional
    :return: Distributions
    :rtype: tuple[np.array]
    """

    if isinstance(blocks, str):
        blocks = pr.read_bed(blocks)

    if isinstance(blocks, PyRanges):
        blocks = blocks.df

    blocks["state"] = blocks.apply(
        lambda row: _determine_state(row[3], row[4], sample_map), axis=1)
    if save_blocks_bed is not None and save_blocks_bed is not False:
        pr.PyRanges(blocks).to_bed(save_blocks_bed)

    s1, s2, s3 = [_get_s_distr(blocks, i) for i in [1, 2, 3]]
    if save_distr_npz is not None and save_distr_npz is not False:
        np.savez(save_distr_npz,
                 s1=s1,
                 s2=s2,
                 s3=s3)

    return s1, s2, s3
