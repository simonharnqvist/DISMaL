from dismal.multimodel import MultiModel
from dismal.divergencemodel import DivergenceModel
import numpy as np

def test_make_model_space():
    multimod = MultiModel(
        deme1_id="pop1",
        deme2_id="pop2",
        allow_asymmetric_migration=True
    )

    model_space = multimod.make_model_space()
    assert isinstance(model_space, list) and isinstance(model_space[0], dict)
    assert len(model_space) == 7

def test_add_model_spec():
    multimod = MultiModel(
        deme1_id="pop1",
        deme2_id="pop2",
        allow_asymmetric_migration=True
    )

    multimod.make_model_space()

    multimod.add_model_spec({"epochs": 3, 
                              "migration": (True, True, False), 
                              "asym_migration": (True, True, False), 
                              "unidirectional_migration": (("pop1", "pop2"), ("pop1", "pop2"))})
    
    assert len(multimod.model_space) == 8

def test_model_output():
    
    multimod = MultiModel(
        deme1_id="pop1",
        deme2_id="pop2",
        allow_asymmetric_migration=True
    )
    
    multimod.fit_models(s1=np.ones(10),
                        s2=np.ones(10),
                        s3=np.ones(10),
                        blocklen=100)
    assert len(multimod.models) == len(multimod.model_space)
    for mod in multimod.models:
        assert isinstance(mod, DivergenceModel)