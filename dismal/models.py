from dismal.demographicmodel import DemographicModel

# TODO: wrapper for forcing unidirectional migration?

def iso_two_epoch(sampled_deme_names=None):
    """Create two-epoch isolation model with default names"""
    mod = DemographicModel(model_ref="2-epoch-ISO")
    mod.add_epoch(n_demes=2,
                    deme_ids=sampled_deme_names,
                    migration=False)
    mod.add_epoch(n_demes=1,
                    migration=False)
    
    return mod

def im(sampled_deme_names=None, asymmetric_migration=True):
    """Create two-epoch isolation-with-migration model with default names."""
    mod = DemographicModel(model_ref="2-epoch-IM")
    mod.add_epoch(n_demes=2,
                    deme_ids=sampled_deme_names,
                    migration=True,
                    asymmetric_migration=asymmetric_migration)
    mod.add_epoch(n_demes=1,
                    migration=False)
    
    return mod

def gim(sampled_deme_names=None, asymmetric_migration=True):
    """Create three-epoch GIM model (allow migration post-split) using default names"""
    mod = DemographicModel(model_ref="3-epoch-GIM")
    mod.add_epoch(n_demes=2,
                    deme_ids=sampled_deme_names,
                    migration=True, asymmetric_migration=asymmetric_migration)
    mod.add_epoch(n_demes=2,
                    migration=True, asymmetric_migration=asymmetric_migration)
    mod.add_epoch(n_demes=1,
                    migration=False)
        
    return mod

def iim(sampled_deme_names=None, asymmetric_migration=True):
    """Create three-epoch isolation-with-initial-migration model (migration only in middle epoch) with default names"""
    mod = DemographicModel(model_ref="3-epoch-IIM")
    mod.add_epoch(n_demes=2,
                  deme_ids=sampled_deme_names,
                  migration=False)
    mod.add_epoch(n_demes=2,
                  migration=True, asymmetric_migration=asymmetric_migration)
    mod.add_epoch(n_demes=1,
                  migration=False)
        
    return mod
    

def secondary_contact(sampled_deme_names=None, asymmetric_migration=True):
    """Create three-epoch secondary contact model (migration only in most recent epoch) with default names"""
    mod = DemographicModel(model_ref="3-epoch-SEC")
    mod.add_epoch(n_demes=2,
                  deme_ids=sampled_deme_names,
                  migration=True, asymmetric_migration=asymmetric_migration)
    mod.add_epoch(n_demes=2,
                  migration=False)
    mod.add_epoch(n_demes=1,
                  migration=False)
        
    return mod
    

def iso_three_epoch(sampled_deme_names=None):
    """Create three-epoch isolation model (no migration) with default names"""
    mod = DemographicModel(model_ref="3-epoch-ISO")
    mod.add_epoch(n_demes=2,
                  deme_ids=sampled_deme_names,
                  migration=False)
    mod.add_epoch(n_demes=2,
                  migration=False)
    mod.add_epoch(n_demes=1,
                  migration=False)
    return mod