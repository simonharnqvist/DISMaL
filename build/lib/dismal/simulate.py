import msprime
import numpy as np
from collections import Counter
from dismal.preprocess import s_matrix_from_dicts

class MsprimeSimulation:

    def __init__(self, theta0, theta1, theta2, theta1_prime, theta2_prime, 
                          t1, v, m1_star, m2_star, m1_prime_star, m2_prime_star, Ne, block_len, num_replicates):
        self.t0_coal_time = (t1+v)
        self.t0_gen_time = self.coal_to_generations(self.t0_coal_time, Ne)
        self.t1_coal_time = t1
        self.t1_gen_time = self.coal_to_generations(self.t1_coal_time, Ne)
        self.block_len = block_len
        self.num_replicates = num_replicates
        self.theta0 = theta0 
        self.theta1 = theta1 
        self.theta2 = theta2 
        self.theta1_prime = theta1_prime 
        self.theta2_prime = theta2_prime 
        self.ms_theta0, self.ms_theta1, self.ms_theta2, self.ms_theta1_prime, self.ms_theta2_prime = self.per_block_to_per_base_theta()
        self.mu_per_bp = (self.ms_theta1/(4*Ne))
        self.Ne_a, self.Ne_b, self.Ne, self.Ne_c1, self.Ne_c2 = self.thetas_to_ne()
        self.m_b2_to_b1 = self.number_to_proportion_migration_rate(m1_star, self.Ne_b)
        self.m_b1_to_b2 = self.number_to_proportion_migration_rate(m2_star, self.Ne)
        self.m_c2_to_c1 = self.number_to_proportion_migration_rate(m1_prime_star, self.Ne_c2)
        self.m_c1_to_c2 = self.number_to_proportion_migration_rate(m2_prime_star, self.Ne_c1)
        self.S = self.simulate()

    def __repr__(self):
        return self.S
    
    def __str__(self):
        return self.S

    @staticmethod
    def coal_to_generations(coal_time, Ne):
        return(2*Ne*coal_time)
    
    def per_block_to_per_base_theta(self):
        return [theta/self.block_len for theta in [self.theta0, self.theta1, self.theta2, self.theta1_prime, self.theta2_prime]]
    
    def thetas_to_ne(self):
        return [theta/(4*self.mu_per_bp) for theta in [self.ms_theta0, self.ms_theta1, self.ms_theta2, self.ms_theta1_prime, self.ms_theta2_prime]]
    
    @staticmethod
    def number_to_proportion_migration_rate(M, Ne):
        """Convert big M to small m, and account for backwards in time"""
        return (M/2)/Ne
    
    def simulate(self):
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=self.Ne_a)
        demography.add_population(name="b1", initial_size=self.Ne)
        demography.add_population(name="b2", initial_size=self.Ne_b)
        demography.add_population(name="c1", initial_size=self.Ne_c1)
        demography.add_population(name="c2", initial_size=self.Ne_c2)
        demography.add_population_split(time=self.t0_gen_time, derived=["b1", "b2"], ancestral="a")
        demography.add_population_split(time=self.t1_gen_time, derived=["c1"], ancestral="b1")
        demography.add_population_split(time=self.t1_gen_time, derived=["c2"], ancestral="b2")
        demography.set_migration_rate("b2", "b1", self.m_b2_to_b1) # NB: backwards in time in msprime
        demography.set_migration_rate("b1", "b2", self.m_b1_to_b2)
        demography.set_migration_rate("c2", "c1", self.m_c2_to_c1)
        demography.set_migration_rate("c1", "c2", self.m_c1_to_c2)
        demography.sort_events()

        ts_state1 = msprime.sim_ancestry(samples={'c1':2, 'c2':0}, demography=demography,
                                      sequence_length=self.block_len, num_replicates=self.num_replicates, ploidy=2)
        ts_state2 = msprime.sim_ancestry(samples={'c1':0, 'c2':2}, demography=demography,
                                      sequence_length=self.block_len, num_replicates=self.num_replicates, ploidy=2)
        ts_state3 = msprime.sim_ancestry(samples={'c1':1, 'c2':1}, demography=demography,
                                      sequence_length=self.block_len, num_replicates=self.num_replicates, ploidy=2)

        s = []
        for ts in [ts_state1, ts_state2, ts_state3]:
            sim = np.zeros(self.num_replicates)
            for replicate_index, ts in enumerate(ts):
                ts_muts = msprime.sim_mutations(ts, rate=self.mu_per_bp, discrete_genome=False)
                sim[replicate_index] = ts_muts.divergence(sample_sets=[[0],[2]], span_normalise=False)

            s_counts_dict = Counter(sim)
            s.append(dict(sorted(s_counts_dict.items())))


        return s_matrix_from_dicts(s)