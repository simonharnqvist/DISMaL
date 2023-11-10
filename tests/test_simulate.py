import pytest
import numpy as np
from numpy.testing import assert_allclose
from dismal.simulate import DemographySimulation

@pytest.fixture()
def setup_simulation():
    return DemographySimulation(
        block_thetas=[3,2,4,3,6],
        epoch_durations=[2,1],
        migration_rates_fraction=[5e-8, 5e-8, 5e-9, 5e-9],
        blocks_per_state = 100
    )


def test_site_theta(setup_simulation):
    assert_allclose(setup_simulation.site_theta, np.array(
          [0.006, 0.004, 0.008, 0.006, 0.012]))


def test_deme_sizes_2N(setup_simulation):
    mutation_rate = 1e-9
    assert_allclose(setup_simulation.deme_sizes_2N,
                    np.array([0.006, 0.004, 0.008, 0.006, 0.012]) / (2 * mutation_rate)) 
        

def test_epoch_durations_generations(setup_simulation):
    assert_allclose(setup_simulation.epoch_durations_generations,
                    np.array([8_000_000, 4_000_000]))
    

def test_split_times_generations(setup_simulation):
    assert_allclose(setup_simulation.split_times_generations,
                    np.array([8_000_000, 12_000_000]))
    

def test_sarrays_correct_length(setup_simulation):
    n_blocks_per_state = 100
    for s in [setup_simulation.s1, setup_simulation.s2, setup_simulation.s3]:
        assert s.shape[0] == n_blocks_per_state 
