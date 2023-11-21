from dismal.callset import CallSet
import pandas as pd
import numpy as np
from pathlib import Path
import pytest

# Setup

@pytest.fixture
def make_callset():
    vcf_path = f"{Path(__file__).parent.parent.resolve()}/test_data/brenthis_500K.vcf.gz"
    vcf_npz = f"{Path(__file__).parent.parent.resolve()}/test_data/vcf.npz"
    return CallSet(vcf_path, vcf_npz)

def test_CallSet_from_vcf(make_callset):
    assert len(make_callset.pos) > 0

def test_CallSet_from_npz(make_callset):
    make_callset = CallSet(npz_path=f"{Path(__file__).parent.parent.resolve()}/test_data/vcf.npz")
    assert len(make_callset.pos) == len(make_callset.pos)
