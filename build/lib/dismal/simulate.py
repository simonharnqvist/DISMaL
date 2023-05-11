import msprime
import numpy as np
from collections import Counter
from dismal.preprocess import s_matrix_from_dicts

def simulate_msprime(theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star, Ne, block_len, num_replicates):

    # Time parameter conversions
    t0_coal_time = (t1+v)
    t0_gen_time = 2*Ne*t0_coal_time
    t1_coal_time = t1
    t1_gen_time = 2*Ne*t1_coal_time

    # Convert theta values from per block to per base
    ms_theta0, ms_theta1, ms_theta2, ms_theta1_prime, ms_theta2_prime = [theta/block_len for theta in [theta0, theta1, theta2, theta1_prime, theta2_prime]]

    # Mutation rate
    mu_per_bp = (ms_theta1/(4*Ne))

    # Convert thetas to Ne
    Ne_a, Ne_b, Ne_c1, Ne_c2 = [theta/(4*mu_per_bp) for theta in [ms_theta0, ms_theta2, ms_theta1_prime, ms_theta2_prime]]

    demography = msprime.Demography()
    demography.add_population(name="a", initial_size=Ne_a)
    demography.add_population(name="b1", initial_size=Ne)
    demography.add_population(name="b2", initial_size=Ne_b)
    demography.add_population(name="c1", initial_size=Ne_c1)
    demography.add_population(name="c2", initial_size=Ne_c2)
    demography.add_population_split(time=t0_gen_time, derived=["b1", "b2"], ancestral="a")
    demography.add_population_split(time=t1_gen_time, derived=["c1"], ancestral="b1")
    demography.add_population_split(time=t1_gen_time, derived=["c2"], ancestral="b2")
    demography.set_migration_rate("b2", "b1", m2_star/2) # NB: backwards in time in msprime
    demography.set_migration_rate("b1", "b2", m1_star/2)
    demography.set_migration_rate("c2", "c1", m2_prime_star/2)
    demography.set_migration_rate("c1", "c2", m1_prime_star/2)
    demography.sort_events()

    ts_state1 = msprime.sim_ancestry(samples={'c1':2, 'c2':0}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=2)
    ts_state2 = msprime.sim_ancestry(samples={'c1':0, 'c2':2}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=2)
    ts_state3 = msprime.sim_ancestry(samples={'c1':1, 'c2':1}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=2)

    s = []
    for ts in [ts_state1, ts_state2, ts_state3]:
        sim = np.zeros(num_replicates)
        for replicate_index, ts in enumerate(ts):
            ts_muts = msprime.sim_mutations(ts, rate=mu_per_bp, discrete_genome=False)
            sim[replicate_index] = ts_muts.divergence(sample_sets=[[0],[2]], span_normalise=False)

        s_counts_dict = Counter(sim)
        s.append(dict(sorted(s_counts_dict.items())))


    return s_matrix_from_dicts(s)