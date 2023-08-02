from dismal.preprocess import gff3_to_blocks, CallSet, block_genotype_array, index_individuals_wo_missing_sites, n_segr_sites, blockwise_segregating_sites, from_vcf_gff3
import numpy as np
import pandas as pd

# Setup
gff3 = "../test_data/brenthis_head1000.gff3"
blocklen = 100
ld_dist = 10000
blocks_df = gff3_to_blocks(gff3, blocklen=blocklen, min_block_distance=ld_dist)
vcf_path = "../test_data/brenthis_500K.vcf.gz"
samples_map = "../test_data/samples_map.csv"
samples = pd.read_csv(samples_map)
vcf_npz = "../test_data/vcf.npz"
callset = CallSet(vcf_path, vcf_npz)

test_gt = np.array(
    
        [
            [
                [0,0],
                [-1, 1],
                [0, 1],
                [1,1],
                [1,1]
            ],
            [
                [0,0],
                [-1, 1],
                [0, 1],
                [1,0],
                [1,0]
            ],
            [
                [0,0],
                [-1, 1],
                [0, -1],
                [1,1],
                [1,1]
            ]
        ]
)


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


def test_CallSet_from_vcf():
    assert len(callset.pos) > 0


def test_CallSet_from_npz():
    callset_npz = CallSet(npz_path=vcf_npz)
    assert len(callset_npz.pos) == len(callset.pos)

def test_block_genotype_array_correct_type():
    block_gt = block_genotype_array(callset, callset.chromosomes[0], blocks_df.loc[0, "start"], blocks_df.loc[0, "end"])
    assert isinstance(block_gt, np.ndarray)

def test_index_individuals_wo_missing_sites():
    np.testing.assert_array_equal(index_individuals_wo_missing_sites(test_gt), [0, 3, 4])

def test_n_segr_sites():
    arr = test_gt[:, (0, 3), 1]
    assert n_segr_sites(arr) == 2

def test_blockwise_segregating_sites_correct_shape():
    s3 = blockwise_segregating_sites(blocks_df, callset, samples)
    assert s3.shape[0] == 3

def test_from_vcf_gff3():
    s3 = from_vcf_gff3(vcf=vcf_path, gff3=gff3, samples_map=samples_map, vcf_npz_path=vcf_npz)
    assert s3.shape[0] == 3

