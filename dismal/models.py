from dismal.demographicmodel import DemographicModel

def iso_two_epoch():
    mod = DemographicModel(model_ref="2-epoch-ISO")
    mod.add_epoch(n_demes=2,
                    migration=False)
    mod.add_epoch(n_demes=1,
                    migration=False)
    
    return mod

def im():
    mod = DemographicModel(model_ref="2-epoch-IM")
    mod.add_epoch(n_demes=2,
                    migration=True)
    mod.add_epoch(n_demes=1,
                    migration=False)
    
    return mod

def gim():
    """Create three-epoch GIM model (allow migration post-split) using default names"""
    mod = DemographicModel(model_ref="3-epoch-GIM")
    mod.add_epoch(n_demes=2,
                    migration=True)
    mod.add_epoch(n_demes=2,
                    migration=True)
    mod.add_epoch(n_demes=1,
                    migration=False)
        
    return mod

def iim():
    """Create three-epoch isolation-with-initial-migration model (migration only in middle epoch) with default names"""
    mod = DemographicModel(model_ref="3-epoch-IIM")
    mod.add_epoch(n_demes=2,
                  migration=False)
    mod.add_epoch(n_demes=2,
                  migration=True)
    mod.add_epoch(n_demes=1,
                  migration=False)
        
    return mod
    

def secondary_contact():
    """Create three-epoch secondary contact model (migration only in most recent epoch) with default names"""
    mod = DemographicModel(model_ref="3-epoch-SEC")
    mod.add_epoch(n_demes=2,
                  migration=True)
    mod.add_epoch(n_demes=2,
                  migration=False)
    mod.add_epoch(n_demes=1,
                  migration=False)
        
    return mod
    

def iso_three_epoch():
    """Create three-epoch isolation model (no migration) with default names"""
    mod = DemographicModel(model_ref="3-epoch-ISO")
    mod.add_epoch(n_demes=2,
                  migration=False)
    mod.add_epoch(n_demes=2,
                  migration=False)
    mod.add_epoch(n_demes=1,
                  migration=False)
    return mod