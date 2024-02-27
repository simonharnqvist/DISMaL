from dismal.callset import CallSet
from archive.blockgt import BlockGT
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

@pytest.fixture
def make_block():
    callset = CallSet(npz_path=f"{Path(__file__).parent.parent.resolve()}/test_data/vcf.npz")
    samples_map = pd.read_csv(f"{Path(__file__).parent.parent.resolve()}/test_data/samples_map.csv")
    pop1_samples = samples_map["sample"][0:5]
    pop2_samples = samples_map["sample"][6:13]
    gt = callset.gt[np.where((callset.chromosomes == 0) 
                                             & (callset.pos >= 50_000) 
                                             & (callset.pos <= 51_000))]

    block = BlockGT(gt=gt, callset_samples=callset.samples, pop1_samples=pop1_samples, pop2_samples=pop2_samples)

    return block

def test_block_gt_shape(make_block):
    assert make_block.gt.shape == (14, 13, 2)

def test_pop_gt_shapes(make_block):
    assert make_block.pop1_gt.shape == (14, 5, 2)
    assert make_block.pop2_gt.shape == (14, 7, 2)

def test_flattened_shape(make_block):
    assert make_block.pop1_gt_flat.shape == (14, 10)

def test_filtered_shape(make_block):
    assert make_block.pop1_gt_filtered.shape == (14, 2)

def test_s1(make_block):
    assert make_block.s1 == [1]

def test_s2(make_block):
    assert make_block.s2 == [6, 3, 3, 9, 9, 0]

def test_s3(make_block):
    assert make_block.s3 == [3, 3, 6, 6, 4, 2, 7, 7]