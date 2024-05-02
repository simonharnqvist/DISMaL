from archive.blocks import gff3_to_blocks
from dismal.callset import CallSet
import pandas as pd
import numpy as np
from pathlib import Path


gff3_path = f"{Path(__file__).parent.parent.resolve()}/test_data/brenthis_head1000.gff3"
blocklen = 100
ld_dist = 10000
blocks_df = gff3_to_blocks(gff3_path, blocklen=blocklen, min_block_distance=ld_dist)

def test_gff3_to_blocks_correct_columns():
    assert (blocks_df.columns == ["chr", "start", "end", "block_id"]).all()


def test_gff3_to_blocks_contains_blocks():
    assert len(blocks_df) > 0


def test_gff3_to_blocks_has_blocks_of_correct_length():
    blocksizes = blocks_df["end"] - blocks_df["start"]
    assert (blocksizes == blocklen).all()


def test_gff3_to_blocks_filters_ld():
    for chr in blocks_df["chr"].unique():
        chr_blocks = blocks_df[blocks_df["chr"] == chr]
        assert (chr_blocks["start"].diff()[1:] > ld_dist + blocklen).all() # zeroth element evals to False since no block before