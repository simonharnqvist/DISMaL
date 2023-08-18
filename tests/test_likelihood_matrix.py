import numpy as np
from dismal.demography import Epoch
from dismal.infer import DivergenceModel
from dismal.likelihood_matrix import LikelihoodMatrix

def test_LikelihoodMatrix_sums_to_rgim_expectations():

    rgim = 87.0
    mod = DivergenceModel(epochs=3, allow_migration=True)
    lm_sum = np.sum(LikelihoodMatrix(params=[3, 2, 4, 3, 6, 8, 4, 0.2, 0.4, 0.04, 0.08], S = np.ones(shape=(3,10)), epoch_objects=mod.epochs).matrix)

    assert np.isclose(rgim, lm_sum)
    