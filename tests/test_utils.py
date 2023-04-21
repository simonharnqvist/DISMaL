from dismal.utils import _model_to_opt_params, _opt_to_model_params

model_params1 = {"a":1.0, "b":1.0, "c1":1.0, "c2":1.0, "tau1":1.0, "tau0":2.0, "m1":0.0, "m2":0.0, "m1_prime":0.0, "m2_prime":0.0, "theta":5.0}
opt_params1 = {"theta0":5.0, "theta1":5.0, "theta2":5.0, "theta1_prime":5.0, "theta2_prime":5.0, "t1":5.0, "v":5.0, "m1_star":0.0, "m2_star":0.0, "m1_prime_star":0.0, "m2_prime_star":0.0}
model_params2 = {"a":1.0, "b":2.0, "c1":1, "c2":1.0, "tau1":1.0, "tau0":2.0, "m1":0.1, "m2":0.1, "m1_prime":0.2, "m2_prime":0.2, "theta":2.0}
opt_params2 = {"theta0":2.0, "theta1":2.0, "theta2":4.0, "theta1_prime":2.0, "theta2_prime":2.0, "t1":2.0, "v":2.0, "m1_star":0.1, "m2_star":0.2, "m1_prime_star":0.2, "m2_prime_star":0.2}

def test_model_to_opt_params_no_migration():
    assert _model_to_opt_params(model_params1) == opt_params1

def test_model_to_opt_params_with_migration():
    assert _model_to_opt_params(model_params2) == opt_params2

def test_opt_to_model_params_no_migration():
    assert _opt_to_model_params(opt_params1) == model_params1

def test_opt_to_model_params_with_migration():
    assert _opt_to_model_params(opt_params2) == model_params2