#from didle.didle import demography

#def test_coal_time_pdf_is_zero_for_state3_if_no_mig():
#    assert all(v == 0 for v in \
#               [demography.coal_time_pdf(t=t, tau0=1, tau1=0.5, a=1.1, b=1.1, c1=1.1, c2=1.1, m1_prime=0, m2_prime=0, m1=0, m2=0, i=3)\
#                for t in [0, 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99]]),\
#        "Between t=0 and t=tau0, all coalescence times for state 3 should have a probability of zero if there is no migration"

#def test_reparameterisation_to_cwh_parameters():
#    assert dem.reparamaterise([1, 1, 1, 1, 1, 2, 0.3, 0.3, 0.3, 0.3, 5]) == [5, 5, 5, 5, 5, 5, 5, 0.3, 0.3, 0.3, 0.3],\
#        ".reparameterise() does not yield correct results"

#def test_reparameterisation_back_to_paper_parameters():
#    assert dem.reparamaterise_back([5, 5, 5, 5, 5, 5, 5, 0.3, 0.3, 0.3, 0.3]) == [1, 1, 1, 1, 1, 2, 0.3, 0.3, 0.3, 0.3, 5]