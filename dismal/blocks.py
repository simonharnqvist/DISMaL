import allel
import pandas as pd
import numpy as np
import tqdm


def _filter_chromosomes(blocks_df, select=None, remove=None):
    """UNTESTED: Select (include only) or remove chromosomes by name"""

    if select is not None:
        assert remove is None, "Cannot use both 'select' and 'remove' parameters"
        chr_filtered_blocks = blocks_df[blocks_df["chr"].isin(select)]
    elif remove is not None:
        assert select is None, "Cannot use both 'select' and 'remove' parameters"
        chr_filtered_blocks = blocks_df[~blocks_df["chr"].isin(select)]
    else:
        chr_filtered_blocks = blocks_df

    return chr_filtered_blocks


def _filter_ld(blocks_df, min_block_distance):
    """Filter blocks for minimum distance between them to reduce effect of LD"""
    filtered_rows = [dict(blocks_df.iloc[0, :])]

    print("Filtering on minimum block distance...")
    for index, row in tqdm.tqdm(blocks_df.iterrows()):
        if (blocks_df.loc[index, "chr"] != filtered_rows[-1]["chr"]): # if different chromosomes, add block
            filtered_rows.append(dict(blocks_df.iloc[index, :]))
        elif (blocks_df.loc[index, "start"] - filtered_rows[-1]["end"] > min_block_distance): # if same chromosome, check distance, add if far enough apart
            filtered_rows.append(dict(blocks_df.iloc[index, :]))
        else:
            pass

    return pd.DataFrame(filtered_rows)

def _trim_partitions(blocks_df, trim_start=0, trim_end=0):
    """UNTESTED: Trim each partition unit by trim_start at beginning and trim_end at end"""
    blocks_df_trimmed = blocks_df
    blocks_df_trimmed["start"] = blocks_df_trimmed["start"] + trim_start
    blocks_df_trimmed["end"] = blocks_df_trimmed["end"] + trim_end

    return blocks_df_trimmed


def _expand_partitions(row, blocklen):
    """Expand a partition (row) to blocks of length blocklen"""
    start = row["start"]
    end = row["end"]
    assert end - start >= blocklen

    start_idxs = list(range(start, end, blocklen+1))
    end_idxs = [i+blocklen for i in start_idxs]

    if end_idxs[-1] > end:  # if last block overshoots
        start_idxs.pop(-1)
        end_idxs.pop(-1)

    partition_blocks = pd.DataFrame(columns=["chr", "start", "end"])
    partition_blocks["chr"] = [row["chr"]] * len(start_idxs)
    partition_blocks["start"] = start_idxs
    partition_blocks["end"] = end_idxs

    return partition_blocks


def gff3_to_blocks(gff3_path, blocklen, 
                   min_block_distance, genomic_partition="intron", 
                   trim_starts=10, trim_ends=10,
                   select_chromosomes=None,
                   exclude_chromosomes=None,
                   parquet_path=None):
    """Subset GFF3 file for intronic blocks of length blocklen, at least ld_dist apart"""

    pd.set_option('chained_assignment', None)

    gff = allel.gff3_to_dataframe(gff3_path)[["seqid", "type", "start", "end"]]
    gff.columns = ["chr", "type", "start", "end"]

    # Filter chromosomes
    gff_filtered_chr = _filter_chromosomes(gff, select=select_chromosomes, remove=exclude_chromosomes)

    # Subset paritions_ regions
    partitions_df = gff_filtered_chr[gff_filtered_chr["type"] == genomic_partition]

    # Subset len(intron) > blocklen
    partitions_min_blocklen = partitions_df[(partitions_df["end"] - partitions_df["start"]) > blocklen]

    # Expand each intron to multiple blocks of length blocklen
    print("Expanding partitions to blocks...")
    blocks_unfiltered = pd.concat([_expand_partitions(partitions_min_blocklen.iloc[row, :],
                                                      blocklen=blocklen) for row in 
                                                      tqdm.tqdm(range(len(partitions_min_blocklen)))])
    blocks_unfiltered = blocks_unfiltered.reset_index(drop=True)

    # remove conserved regions at 
    blocks_trimmed = _trim_partitions(blocks_unfiltered, 
                                      trim_start=trim_starts, trim_end=trim_ends)

    # Filter for LD
    blocks_ld_filtered = _filter_ld(blocks_trimmed, min_block_distance)

    # Include block information in output df
    blocks_ld_filtered.loc[:, "block_id"] = blocks_ld_filtered.loc[:, "chr"] \
        + ":" + blocks_ld_filtered.loc[:, "start"].astype(str) + "-" \
            + blocks_ld_filtered.loc[:, "end"].astype(str)

    blocks = blocks_ld_filtered

    blocks["chr"] = pd.Categorical(blocks["chr"]).codes

    if parquet_path is not None:
        blocks.to_parquet(parquet_path)

    return blocks


def blocks_from_parquet(parquet_path):
    """Read blocks from parquet path"""
    return pd.read_parquet(parquet_path)
