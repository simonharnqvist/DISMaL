import pytest
import numpy as np
from numpy.testing import assert_allclose
from dismal.simulate import DemographySimulation

@pytest.fixture()
def setup_simulation():
    return DemographySimulation(
        block_thetas=[3,2,4,3,6],
        epoch_durations=[2,1],
        migration_rates=[0.3, 0.2, 0.2, 0.3],
        blocks_per_state = 100
    )


def test_site_theta(setup_simulation):
    assert_allclose(setup_simulation.site_theta, np.array(
          [0.006, 0.004, 0.008, 0.006, 0.012]))


def test_deme_sizes(setup_simulation):
    mutation_rate = 1e-9
    assert_allclose(setup_simulation.deme_sizes,
                    np.array([0.006, 0.004, 0.008, 0.006, 0.012]) / (4 * mutation_rate)) 
        

def test_epoch_durations_generations(setup_simulation):
    assert_allclose(setup_simulation.epoch_durations_generations,
                    np.array([4_000_000, 2_000_000]))
    

def test_split_times_generations(setup_simulation):
    assert_allclose(setup_simulation.split_times_generations,
                    np.array([4_000_000, 6_000_000]))