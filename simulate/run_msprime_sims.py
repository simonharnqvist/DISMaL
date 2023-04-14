from msprime_simulate import simulate_blocks

# ISO
simulate_blocks(a=1,b=1,c1=1,c2=1,tau1=1,tau0=2,m1=0,m2=0,m1_prime=0,m2_prime=0,theta=0.1,N=1000,n_samples=[6,6], n_blocks=10_000, block_len=100, output_prefix="../../sims_isolation_1/")
