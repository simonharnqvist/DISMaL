import msprime
from dismal import preprocess

def _coalescent_time_to_generations(t, N):
    return t*2*N

def _calculate_Ne(theta, mu):
    return theta/(4*mu)

def msprime_simulate(theta0, theta1, theta2, theta1_prime, theta2_prime, t1, v,
                  m1_star, m2_star, m1_prime_star, m2_prime_star, mutation_rate, num_replicates, block_len):
    demography = msprime.Demography()
    demography.add_population(name="a", initial_size=_calculate_Ne(theta0, mutation_rate))
    demography.add_population(name="b1", initial_size=_calculate_Ne(theta1, mutation_rate))
    demography.add_population(name="b", initial_size=_calculate_Ne(theta2, mutation_rate))
    demography.add_population(name="c1", initial_size=_calculate_Ne(theta1_prime, mutation_rate))
    demography.add_population(name="c2", initial_size=_calculate_Ne(theta2_prime, mutation_rate))

    N = _calculate_Ne(theta1, mutation_rate)
    
    tau0 = t1/theta1
    tau1 = (t1+v)/theta1

    demography.add_population_split(time=_coalescent_time_to_generations((tau0), N), derived=["b1", "b"], ancestral="a")
    demography.add_population_split(time=_coalescent_time_to_generations(tau1, N), derived=["c1"], ancestral="b1")
    demography.add_population_split(time=_coalescent_time_to_generations(tau1, N), derived=["c2"], ancestral="b")
    demography.set_migration_rate("b1", "b", m1_star)
    demography.set_migration_rate("b", "b1", m2_star)
    demography.set_migration_rate("c1", "c2", m1_prime_star)
    demography.set_migration_rate("c2", "c1", m2_prime_star)
    demography.sort_events()

    ts_state1 = msprime.sim_ancestry(samples={'c1':2, 'c2':0}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=2)
    ts_state2 = msprime.sim_ancestry(samples={'c1':0, 'c2':2}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=2)
    ts_state3 = msprime.sim_ancestry(samples={'c1':1, 'c2':1}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=2)

    s = []

    s_dicts = []
    for ts_state in [ts_state1, ts_state2, ts_state3]:
        s = []
        for idx, ts in enumerate(ts_state):
            mts = msprime.sim_mutations(ts, rate=mutation_rate, discrete_genome=False)
            mts.divergence(sample_sets=[[0], [2]], span_normalise=False)
            
        s_dicts.append(preprocess.counts_to_dict(s))

    return s_dicts

def msprime_simulate_old_params(a,b,c1,c2,tau1,tau0,m1,m2,m1_prime,m2_prime,theta,N, num_replicates, block_len):
    demography = msprime.Demography()
    demography.add_population(name="a", initial_size=a*N)
    demography.add_population(name="b1", initial_size=N)
    demography.add_population(name="b", initial_size=b*N)
    demography.add_population(name="c1", initial_size=c1*N)
    demography.add_population(name="c2", initial_size=c2*N)
    demography.add_population_split(time=_coalescent_time_to_generations(tau0, N), derived=["b1", "b"], ancestral="a")
    demography.add_population_split(time=_coalescent_time_to_generations(tau1, N), derived=["c1"], ancestral="b1")
    demography.add_population_split(time=_coalescent_time_to_generations(tau1, N), derived=["c2"], ancestral="b")
    demography.set_migration_rate("b1", "b", m1)
    demography.set_migration_rate("b", "b1", m2)
    demography.set_migration_rate("c1", "c2", m1_prime)
    demography.set_migration_rate("c2", "c1", m2_prime)
    demography.sort_events()

    ts_state1 = msprime.sim_ancestry(samples={'c1':2, 'c2':0}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=1)
    ts_state2 = msprime.sim_ancestry(samples={'c1':0, 'c2':2}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=1)
    ts_state3 = msprime.sim_ancestry(samples={'c1':1, 'c2':1}, demography=demography, sequence_length=block_len, num_replicates=num_replicates, ploidy=1)

    mutation_rate = theta/(4*N)
    s = []

    s_dicts = []
    for ts_state in [ts_state1, ts_state2, ts_state3]:
        s = []
        for ts in ts_state:
            mts = msprime.sim_mutations(ts, rate=mutation_rate, discrete_genome=False)
            s.append(mts.get_num_mutations())
        s_dicts.append(preprocess.counts_to_dict(s))

    return s_dicts






 
# def _simulate_block(a,b,c1,c2,tau1,tau0,m1,m2,m1_prime,m2_prime,theta,N, n_samples, block_len):
#     demography = msprime.Demography()
#     demography.add_population(name="a", initial_size=a*N)
#     demography.add_population(name="b1", initial_size=N)
#     demography.add_population(name="b", initial_size=b*N)
#     demography.add_population(name="c1", initial_size=c1*N)
#     demography.add_population(name="c2", initial_size=c2*N)
#     demography.add_population_split(time=_coalescent_time_to_generations(tau0, N), derived=["b1", "b"], ancestral="a")
#     demography.add_population_split(time=_coalescent_time_to_generations(tau1, N), derived=["c1"], ancestral="b1")
#     demography.add_population_split(time=_coalescent_time_to_generations(tau1, N), derived=["c2"], ancestral="b")
#     demography.set_migration_rate("b1", "b", m1)
#     demography.set_migration_rate("b", "b1", m2)
#     demography.set_migration_rate("c1", "c2", m1_prime)
#     demography.set_migration_rate("c2", "c1", m2_prime)
#     demography.sort_events()
#     ts = msprime.sim_ancestry(samples={'c1':n_samples[0], 'c2':n_samples[1]}, demography=demography, sequence_length=block_len)
#     mutation_rate = _theta_to_mu(theta, N)
#     mts = msprime.sim_mutations(ts, rate=mutation_rate, discrete_genome=False)

#     return mts

# def simulate_blocks_vcf(a,b,c1,c2,tau1,tau0,m1,m2,m1_prime,m2_prime,theta,N, n_samples, block_len, n_blocks, output_prefix):

#     for i in tqdm.tqdm(range(0, n_blocks)):
#         treeseq = _simulate_block(a,b,c1,c2,tau1,tau0,m1,m2,m1_prime,m2_prime,theta,N, n_samples, block_len)
#         vcf_path = f"{output_prefix}_{i}.vcf"

#         with open(vcf_path, 'w') as f:
#             treeseq.write_vcf(f)

#     params_path = f"{output_prefix}.params.txt"
#     with open(params_path, 'w') as f:
#         f.write(f"a:{a}, b:{b}, c1:{c1}, c2:{c2}, tau1:{tau1}, tau0:{tau0}, theta:{theta}, m1:{m1}, m2:{m2}, m1_prime;{m1_prime}, m2_prime:{m2_prime}, seq_len:{block_len}, N:{N}")
#         print(f"Parameters written to {params_path}")

