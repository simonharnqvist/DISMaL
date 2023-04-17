from dismal.likelihood import _expected_gs_u, _lambda_summation, _pdf_gs, _pdf_s, _sval_likelihood
from math import exp

def test_expected_gs_u_evals_cf_matlab():
    assert _expected_gs_u(s=1, theta=1, lmbda=1) == 0.25
    assert _expected_gs_u(s=5, theta=1, lmbda=1) == 0.015625
    assert round(_expected_gs_u(s=1, theta=5, lmbda=1), 4) == 0.1389
    assert round(_expected_gs_u(s=1, theta=1, lmbda=5), 4) == 0.1389
    assert _expected_gs_u(s=0, theta=1, lmbda=1) == 0.5

def test_lambda_summation_evals_cf_matlab():
    assert _lambda_summation(l=0, theta=1, lmbda=1, t=1) == 1
    assert _lambda_summation(l=1, theta=1, lmbda=1, t=1) == 3
    assert round(_lambda_summation(l=3, theta=1, lmbda=1, t=1), 4) == 6.3333
    assert _lambda_summation(l=1, theta=5, lmbda=1, t=1) == 7
    assert _lambda_summation(l=1, theta=1, lmbda=1, t=5) == 11

def test_pdf_gs():
    assert _pdf_gs(s=0, theta=1, lmbda=1, t=1) == 1/2 * exp(-1)
    assert _pdf_gs(s=1, theta=1, lmbda=1, t=1) == 3/4 * exp(-1)

def test_pdf_s_matches_theory_state1_state2():
    def _expect(k, theta):
        return (1/(theta+1)) * (theta/(theta+1))**k
    
    for s in [0, 1, 2, 5, 10, 50]:
        p_s = _pdf_s(s=s, a=1, b=1, c1=1, c2=1, tau1=1, tau0=2, m1=0, m2=0, m1_prime=0, m2_prime=0, theta=5, state=1)
        expectation = _expect(s, 5)
        assert round(p_s, 5) == round(expectation, 5)

def test_pdf_s_state2_mirrors_state1():
    for s in [0, 1, 2, 5, 10, 50]:
        p_s1 = _pdf_s(s=s, a=1, b=1, c1=1, c2=1, tau1=1, tau0=2, m1=0, m2=0, m1_prime=0, m2_prime=0, theta=5, state=1)
        p_s2 = _pdf_s(s=s, a=1, b=1, c1=1, c2=1, tau1=1, tau0=2, m1=0, m2=0, m1_prime=0, m2_prime=0, theta=5, state=2)
        assert p_s1 == p_s2

def test_sval_likelihood_count1_equal_to_pdf_s():
    assert all([_sval_likelihood(s_val=s, s_count=1, params=[1,1,1,1,1,2,0,0,0,0,5],
                                  state=1) == _pdf_s(s=s, a=1, b=1, c1=1, c2=1, tau1=1,
                                                     tau0=2, m1=0, m2=0, m1_prime=0, m2_prime=0, theta=5, state=1)
                                                       for s in [0, 1, 10, 50]])

def test_sval_likelihood_count_is_multiplier_of_pdf_s():
    for s in [0, 1, 10, 50]:
        p_s = _pdf_s(s=s, a=1, b=1, c1=1, c2=1, tau1=1, tau0=2, m1=0, m2=0, m1_prime=0, m2_prime=0, theta=5, state=1)
        assert _sval_likelihood(s_val=s, s_count=10, params=[1,1,1,1,1,2,0,0,0,0,5], state=1) == 10 * p_s


