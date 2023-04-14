import sys
sys.path.append("..")
from src.matrices import GeneratorMatrix, TransitionMatrix
import numpy as np

def test_q1_sums_to_zero():

    popsizes = [0.01, 0.1, 1, 10, 100]
    mig_rates = [0, 0.0001, 0.01, 1, 10]

    for popsize1 in popsizes:
        for popsize2 in popsizes:
            for mig_rate1 in mig_rates:
                for mig_rate2 in mig_rates:
                    assert int(GeneratorMatrix(matrix_type="Q1", m1_prime=mig_rate1, m2_prime=mig_rate2, c1=popsize1, c2=popsize2).sum()) == 0

def test_q2_sums_to_zero():

    popsizes = [0.01, 0.1, 1, 10, 100]
    mig_rates = [0, 0.0001, 0.01, 1, 10]

    for popsize1 in popsizes:
            for mig_rate1 in mig_rates:
                for mig_rate2 in mig_rates:
                    assert int(GeneratorMatrix(matrix_type="Q2", m1=mig_rate1, m2=mig_rate2, b=popsize1).sum()) == 0

def test_q3_sums_to_zero():

    popsizes = [0.01, 0.1, 1, 10, 100]
    mig_rates = [0, 0.0001, 0.01, 1, 10]

    for popsize1 in popsizes:
        assert int(GeneratorMatrix(matrix_type="Q3", a=popsize1).sum()) == 0


def test_transition_matrix_sums_to_four():
    popsizes = [0.01, 0.1, 1, 10, 100]
    mig_rates = [0, 0.0001, 0.01, 1, 10]
    t_vals = [0, 0.01, 0.1, 0.5, 1, 2, 10]

    for a in popsizes:
        for b in popsizes:
            for c1 in popsizes:
                for c2 in popsizes:
                    for m1 in mig_rates:
                        for m2 in mig_rates:
                            for m1_prime in mig_rates:
                                for m2_prime in mig_rates:
                                    for t in t_vals:
                                        for tau1 in t_vals:
                                            for tau0 in t_vals:
                                                if tau0 > tau1:
                                                    q1 = GeneratorMatrix(matrix_type="Q1", c1=c1, c2=c2, m1_prime=m1_prime, m2_prime=m2_prime)
                                                    q2 = GeneratorMatrix(matrix_type="Q2", b=b, m1=m1, m2=m2)
                                                    q3 = GeneratorMatrix(matrix_type="Q3", a=a)
                                                    assert round(TransitionMatrix(q1=q1, q2=q2, q3=q3,t=t, tau1=tau1, tau0=tau0).sum()) == 4
                                    