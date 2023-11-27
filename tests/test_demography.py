from dismal.demography import Epoch

def test_Epoch_with_no_migration():
    epoch_no_mig = Epoch(n_demes=2,
                 migration=False)
    assert epoch_no_mig.n_mig_params == 0 and epoch_no_mig.asymmetric_migration is False

def test_Epoch_with_symmetric_migration():
    epoch_sym = Epoch(n_demes=2,
                          migration=True,
                            asymmetric_migration=False)
    assert epoch_sym.n_mig_params == 1 and epoch_sym.asymmetric_migration is False

def test_Epoch_with_directional_migration():
    epoch_dir_mig = Epoch(n_demes=2,
                          deme_ids=("pop1", "pop2"),
                          migration=True,
                          asymmetric_migration=False,
                          migration_direction=("pop1", "pop2"))
    assert epoch_dir_mig.n_mig_params == 1 and epoch_dir_mig.migration_direction == ("pop1", "pop2")

