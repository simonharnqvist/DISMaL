from pyranges import PyRanges
import pyranges as pr
import itertools
import pandas as pd
import tqdm
import numpy as np

def get_callable_per_sample(callable):
    """Make PyRanges/BED of callable regions for each sample from mosdepth type PyRanges."""
    callable_df = callable.df.drop("Name", axis=1)
    callable_df["Score"] = callable_df["Score"].str.split(",")
    callable_df = callable_df.explode("Score").rename({"Score":"Sample"})
    return PyRanges(callable_df)

def subset_features(annotation, features):
    """Subset annotation PyRanges for feature name e.g. 'intron'"""
    return PyRanges(annotation.df[annotation.df["Feature"].isin(features)])

def trim_partitions(ranges, trim_start=0, trim_end=0):
    """UNTESTED: Trim each partition unit by trim_start at beginning and trim_end at end"""
    blocks_df_trimmed = ranges.df
    blocks_df_trimmed["start"] = blocks_df_trimmed["start"] + trim_start
    blocks_df_trimmed["end"] = blocks_df_trimmed["end"] + trim_end

    return PyRanges(blocks_df_trimmed)

def subset_ranges_per_sample(ranges):
    """Subset PyRanges for each sample to generate list of PyRanges objects. Sample column must be 'Score'."""
    return [PyRanges(ranges.df[ranges.df["Score"] == sample]) for sample in set(ranges.df["Score"])]

def intersect_samples(sample_ranges): 
    """Intersect all samples with one another to find overlapping regions."""
    res = []
    for (sample1_idx, sample2_idx) in list(itertools.combinations(range(len(sample_ranges)), 2)):
        sample_intersect = sample_ranges[sample1_idx].intersect(sample_ranges[sample2_idx]).df
        merged = PyRanges(sample_intersect).merge(slack=1).df
        merged["sample1"] = sample_intersect["Score"]
        merged["sample2"] = sample_ranges[sample2_idx].df["Score"][0]
        res.append(merged)
    return PyRanges(pd.concat(res))

def make_blocks(ranges, block_size):
    """Partition PyRanges object into blocks of size block_size."""
    windowed = ranges.window(window_size=block_size)
    windowed.Length = windowed.End - windowed.Start 
    return windowed[windowed.Length >= block_size]

def filter_ld(blocks, min_block_distance):
    """Filter blocks for minimum distance between them to reduce effect of LD"""
    filtered_rows = [dict(blocks.iloc[0, :])]

    print("Filtering on minimum block distance...")
    for index, row in tqdm.tqdm(blocks.iterrows()):
        if (blocks.loc[index, "Chromosome"] != filtered_rows[-1]["Chromosome"]): # if different chromosomes, add block
            filtered_rows.append(dict(blocks.iloc[index, :]))
        elif (blocks.loc[index, "Start"] - filtered_rows[-1]["End"] > min_block_distance): # if same chromosome, check distance, add if far enough apart
            filtered_rows.append(dict(blocks.iloc[index, :]))
        else:
            pass

    return pd.DataFrame(filtered_rows)


def make_random_blocks(callset, block_size, chrom_sizes=None, blocks_per_pair=1000):
    """Generate random blocks without considering annotation or coverage/callability"""
    pairs = list(itertools.combinations(callset.samples, 2))
    n_blocks = blocks_per_pair * len(pairs)

    if chrom_sizes is None:
        chrom_sizes = pr.from_dict({
            "Chromosome": [chrom for chrom in set(callset.chromosomes)], 
            "Start":[callset.pos[callset.chromosomes == chrom].min() for chrom in set(callset.chromosomes)], 
            "End": [callset.pos[callset.chromosomes == chrom].max() for chrom in set(callset.chromosomes)]})


    rand_blocks = pr.random(n=n_blocks, length=block_size, 
                     chromsizes=chrom_sizes, strand=False).df
    rand_blocks["sample1"] = [pair[0] for pair in pairs] * blocks_per_pair
    rand_blocks["sample2"] = [pair[1] for pair in pairs] * blocks_per_pair

    return PyRanges(rand_blocks)


def make_blocks(block_size, annotation=None, callable=None, features=None, trim_start=10, trim_end=10):
    """_summary_

    Args:
        block_size (_type_): _description_
        annotation (_type_, optional): _description_. Defaults to None.
        callable (_type_, optional): _description_. Defaults to None.
        features (_type_, optional): _description_. Defaults to None.
        trim_start (int, optional): _description_. Defaults to 10.
        trim_end (int, optional): _description_. Defaults to 10.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    if callable is not None:
        if type(callable) == str:
            callable = pr.read_bed(callable)
        
        callable_per_sample = get_callable_per_sample(callable)
    
    if annotation is not None:
        if type(annotation) == str:
            annotation = pr.read_gff3(annotation)
        
        if features is None:
            features = ["intron"]

        subset_annotation = subset_features(annotation, features)
        trimmed_subset_annotation = trim_partitions(subset_annotation, trim_start=trim_start, trim_end=trim_end)

    if annotation is not None and callable is not None:
        callable_annotation = callable_per_sample.intersect(trimmed_subset_annotation)
    elif annotation is not None:
        callable_annotation = annotation
    elif callable is not None:
        callable_annotation = callable_per_sample
    else:
        raise ValueError("Either an annotation or a callability BED/ranges or both must be provided. Otherwise, use make_random_blocks()")
    

    sample_ranges = subset_ranges_per_sample(callable_annotation)
    intersected = intersect_samples(sample_ranges)
    blocks = make_blocks(intersected, block_size)

    return blocks

def num_seg_sites(callset, chrom_idx, start, stop, sample1, sample2):
    """_summary_

    Args:
        callset (_type_): _description_
        chrom_idx (_type_): _description_
        start (_type_): _description_
        stop (_type_): _description_
        sample1 (_type_): _description_
        sample2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    sample_idxs = np.where(callset.samples == sample1)[0][0], np.where(callset.samples == sample2)[0][0]
    hap_idx = np.random.default_rng().integers(0,2)
    haps = callset.gt[(callset.chromosomes == chrom_idx) & (callset.pos >= start) & (callset.pos < stop)]
    hap1 = haps[:, sample_idxs[0], hap_idx]
    hap2 = haps[:, sample_idxs[1], hap_idx]

    arr = np.array([hap1, hap2])
    arr = arr[:, ~(arr == -1).any(axis=0)]

    return np.sum(arr[0] != arr[1])

def blockwise_seg_sites(blocks, callset):
    """_summary_

    Args:
        blocks (_type_): _description_
        callset (_type_): _description_

    Returns:
        _type_: _description_
    """
    return blocks.df.apply(lambda row: num_seg_sites(callset, 0, row[1], row[2], row[3], row[4]), 
             axis=1)

    

    
    
