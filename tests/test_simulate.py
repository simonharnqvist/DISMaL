import pytest
import numpy as np
from numpy.testing import assert_allclose
from dismal.model_instance import ModelInstance
from dismal.models import gim

@pytest.fixture()
def setup_simulation():
    epochs = gim().epochs
    modinst = ModelInstance([3, 2, 4, 3, 6, 2, 1, 0.3, 0.2, 0.2, 0.3], epochs)
    sim = modinst.simulate(mutation_rate=5e-9, blocklen=200, recombination_rate=0, blocks_per_state=100)
    return sim.demography

def test_deme_sizes(setup_simulation):
    popsizes = [setup_simulation.populations[i].initial_size 
                for i in range(0, len(setup_simulation.populations))]
    assert_allclose(popsizes, np.array(
          [7.5e5, 5e5, 1e6, 7.5e5, 1.5e6]))


def test_split_times_generations(setup_simulation):
    times = sorted(list(set([setup_simulation.events[i].time 
                             for i in range(0, len(setup_simulation.events))])))

    assert_allclose(times,
                    np.array([4e6, 6e6]))
