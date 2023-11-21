from dismal.DemographicModel import DemographicModel

def three_epoch_gim():
    """Create three-epoch GIM model (allow migration post-split) using default names"""
    mod = DemographicModel(model_ref="3-epoch-GIM")
    mod.add_epoch(deme_ids=("pop1", "pop2"),
                    migration=True)
    mod.add_epoch(deme_ids=("pop1_anc", "pop2_anc"),
                    migration=True)
    mod.add_epoch(deme_ids=("ancestral", ),
                    migration=False)
        
    return mod

def three_epoch_iim():
    """Create three-epoch isolation-with-initial-migration model (migration only in middle epoch) with default names"""
    mod = DemographicModel(model_ref="3-epoch-IIM")
    mod.add_epoch(deme_ids=("pop1", "pop2"),
                  migration=False)
    mod.add_epoch(deme_ids=("pop1_anc", "pop2_anc"),
                  migration=True)
    mod.add_epoch(deme_ids=("ancestral", ),
                  migration=False)
        
    return mod
    

def three_epoch_sec():
    """Create three-epoch secondary contact model (migration only in most recent epoch) with default names"""
    mod = DemographicModel(model_ref="3-epoch-SEC")
    mod.add_epoch(deme_ids=("pop1", "pop2"),
                  migration=True)
    mod.add_epoch(deme_ids=("pop1_anc", "pop2_anc"),
                  migration=False)
    mod.add_epoch(deme_ids=("ancestral", ),
                  migration=False)
        
    return mod
    

def three_epoch_iso():
    """Create three-epoch isolation model (no migration) with default names"""
    mod = DemographicModel(model_ref="3-epoch-ISO")
    mod.add_epoch(deme_ids=("pop1", "pop2"),
                  migration=False)
    mod.add_epoch(deme_ids=("pop1_anc", "pop2_anc"),
                  migration=False)
    mod.add_epoch(deme_ids=("ancestral", ),
                  migration=False)
    return mod



def from_dict_spec(model_spec):
    """Generate DemographicModel object from dictionary specification.

    Args:
        model_spec (dict): Dictionary with keys
          "deme_ids", "model_ref", "epochs", "migration", "migration_direction", "asym_migration".
          Example: {"model_ref": "gim_symmetric", "deme_ids": self.deme_ids, "epochs": 3, "migration": (True, True, False), "asym_migration": (False, False, False)}])

    Returns:
        DemographicModel: Model object with specified settings.
    """
    deme_ids = model_spec["deme_ids"]

    if "model_ref" in model_spec.keys():
        model_ref = model_spec["model_ref"]
    else:
        model_ref = None

    mod = DemographicModel(model_ref=model_ref)
    mod.deme_ids = deme_ids

    for epoch_idx in range(model_spec["epochs"]):
        allow_migration_epoch = model_spec["migration"][epoch_idx]
        assert isinstance(allow_migration_epoch, bool)

        if allow_migration_epoch:

            if "asym_migration" in model_spec.keys():
                allow_asymmetric_migration_epoch = model_spec["asym_migration"][epoch_idx]
            else:
                allow_asymmetric_migration_epoch = False

            if "migration_direction" in model_spec.keys():
                migration_direction_epoch = model_spec["migration_direction"][epoch_idx]
                assert len(migration_direction_epoch) == 2, "Migration direction must be specified as (source, target) tuple per epoch"

                assert [migration_direction_epoch[deme_idx] in deme_ids[epoch_idx] for deme_idx in [0,1]]
            else:
                migration_direction_epoch = None

        else:
            allow_asymmetric_migration_epoch = False
            migration_direction_epoch = None

        mod.add_epoch(deme_ids = deme_ids[epoch_idx],
                        migration = allow_migration_epoch,
                        asymmetric_migration = allow_asymmetric_migration_epoch,
                        migration_direction = migration_direction_epoch)

    return mod
