from dismal import infer


mod_2epoch = infer.DivergenceModel(epochs_post_div=1, allow_migration=True, symmetric_migration=True, migration_direction=None)
mod_3epoch = infer.DivergenceModel(epochs_post_div=2, allow_migration=True, symmetric_migration=True, migration_direction=None)

def test_get_params_correct_thetas_2epoch():
    expected_thetas = [("theta_AB", ), ("theta_A1", "theta_B1")]
    observed_thetas = mod_2epoch.get_params()[0]
    assert expected_thetas == observed_thetas

def test_get_params_correct_thetas_3epoch():
    expected_thetas = [("theta_AB", ), ("theta_A1", "theta_B1"), ("theta_A2", "theta_B2")]
    observed_thetas = mod_3epoch.get_params()[0]
    assert expected_thetas == observed_thetas

def test_get_params_correct_taus():
    taus_2epoch, taus_3epoch = mod_2epoch.get_params()[1], mod_3epoch.get_params()[1]
    assert taus_2epoch == ["tau0"] and taus_3epoch == ["tau0", "tau1"]

def test_get_params_correct_Ms_no_migration():
    mod_nomig = infer.DivergenceModel(epochs_post_div=1, allow_migration=False)
    expected_Ms = []
    observed_Ms = mod_nomig.get_params()[2]
    assert expected_Ms == observed_Ms

def test_get_params_correct_Ms_2_epoch_symmetric_mod():
    expected_Ms = [("M1", )]
    observed_Ms = mod_2epoch.get_params()[2]
    assert expected_Ms == observed_Ms

def test_get_params_correct_Ms_3_epoch_symmetric_mod():
    expected_Ms = [("M1", ), ("M2", )]
    observed_Ms = mod_3epoch.get_params()[2]
    assert expected_Ms == observed_Ms

def test_get_params_correct_Ms_3_epoch_asymmetric_mod():

    mod_asymm_flip = infer.DivergenceModel(epochs_post_div=2, allow_migration=True, symmetric_migration=False, migration_direction=[("A1", "B1"), ("B2", "A2")])
    expected_Ms = [("M_A1_B1", ), ("M_B2_A2", )]
    observed_Ms = mod_3epoch.get_params()[2]
    assert expected_Ms == observed_Ms

def test_migration_rate_tuples():
    assert 1 == 0

def test_convert_to_gim_params():
    assert 1 == 0

def test_get_initial_values():
    assert 1 == 0

def test_get_bounds():
    assert 1 == 0

def test_fit():
    assert 1 == 0

def test_composite_likelihood():
    assert 1 == 0
