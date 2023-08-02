import numpy as np
import random

cimport numpy as cnp
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t


cpdef block_gt(cnp.ndarray[cnp.int8_t, ndim=3] gt,
             cnp.ndarray[cnp.int32_t, ndim=1] snp_pos,
             int block_start,
             int blocklen,
             int chrom,
             cnp.ndarray[DTYPE_t, ndim=1] chroms):
    """GT array for a block."""

    cdef block_end = block_start + blocklen
    snps_idxs = np.where((chroms == chrom) & (snp_pos >= block_start) & (snp_pos <= block_end))

    return gt[snps_idxs]


cpdef index_individuals_wo_missing_sites(cnp.ndarray block_gt):
    """Return indices of individuals that have no missing sites in a block"""
    idxs_nonmissing = [np.where([(block_gt[:, i] >= 0).all() 
                                 for i in range(len(block_gt[0]))])]
    return idxs_nonmissing[0][0]

cpdef sample_two_haps(cnp.ndarray[cnp.int8_t, ndim=3] block_gt,
                    cnp.ndarray[DTYPE_t, ndim=1] pop1_idxs,
                    cnp.ndarray[DTYPE_t, ndim=1] pop2_idxs,
                    int state):
    """Sample two haplotypes from a genotype array"""

    rng = np.random.default_rng()
    if state == 1:
        samples_idxs = rng.choice(pop1_idxs, 2, replace=False)
    elif state == 2:
        samples_idxs = rng.choice(pop2_idxs, 2, replace=False)
    else:
        assert state == 3
        samples_idxs = np.array([rng.choice(pop1_idxs, 1, replace=False),
                                  rng.choice(pop2_idxs, 1, replace=False)]).flatten()

    cdef int ploidy = len(block_gt[0][0])
    cdef int haplotype_idx = rng.choice(range(0,ploidy))

    cdef cnp.ndarray samples_gt = block_gt[:, samples_idxs, haplotype_idx]

    return samples_gt
    

cpdef n_segr_sites(cnp.ndarray[cnp.int8_t, ndim=2] samples_gt):
    """Count number of segregating sites between two sampled haplotype blocks."""
    assert (samples_gt >= 0).all(), "Missing sites detected; please remove them before calculating segregating sites."
    cdef int s = np.sum(samples_gt[:, 0] != samples_gt[:, 1])
    return s

cpdef block_segregating_sites(int block_start_idx,
                                int chromosome,
                                int blocklen,
                                cnp.ndarray[cnp.int8_t, ndim=3] gt, 
                                cnp.ndarray [DTYPE_t, ndim=1] chroms,
                                cnp.ndarray [cnp.int32_t, ndim=1] snp_pos,
                                cnp.ndarray[DTYPE_t, ndim=1] pop1_idxs,
                                 cnp.ndarray[DTYPE_t, ndim=1] pop2_idxs,
                                   cnp.ndarray sampling_probs=np.array([0.25, 0.25, 0.50])):

    rng = np.random.default_rng()
    state = rng.choice([1,2,3], 1, p=sampling_probs)[0]
    block = block_gt(gt, snp_pos, block_start_idx, 
            blocklen, chromosome, chroms)

    if len(block) <= 0 or block is None:
        return (None, state)

    indivs_wo_missing_sites = index_individuals_wo_missing_sites(block)
    pop1_nonmissing_idxs = np.intersect1d(pop1_idxs, indivs_wo_missing_sites)
    pop2_nonmissing_idxs = np.intersect1d(pop2_idxs, indivs_wo_missing_sites)


    if state == 1 and len(pop1_nonmissing_idxs) < 2:
        return (None, state)
    elif state == 2 and len(pop2_nonmissing_idxs) < 2:
        return (None, state)
    elif state == 3 and len(pop1_nonmissing_idxs) < 1 or len(pop2_nonmissing_idxs) < 1:
        return (None, state)
    else:
        pass

    samples_gt = sample_two_haps(block, pop1_nonmissing_idxs, pop2_nonmissing_idxs, state=state)
    s = n_segr_sites(samples_gt)

    return s, state


cpdef blockwise_segregating_sites(
    cnp.ndarray[cnp.int64_t, ndim=1] block_start_idxs,
    int blocklen,
    cnp.ndarray[cnp.int8_t, ndim=3] gt, 
    cnp.ndarray [DTYPE_t, ndim=1] chroms,
    cnp.ndarray [cnp.int32_t, ndim=1] snp_pos,
    cnp.ndarray[DTYPE_t, ndim=1] pop1_idxs,
    cnp.ndarray[DTYPE_t, ndim=1] pop2_idxs,
    cnp.ndarray sampling_probs=np.array([0.25, 0.25, 0.50])):
    
    S = np.zeros(shape=(3, len(block_start_idxs)))
    
    for block_idx in range(len(block_start_idxs)):
        s, state = block_segregating_sites(block_start_idxs[block_idx],
        chromosome=chroms[block_idx],
        blocklen=blocklen, 
        gt = gt, 
        chroms=chroms,
        snp_pos = snp_pos,
        pop1_idxs=pop1_idxs,
        pop2_idxs=pop2_idxs,
        sampling_probs=sampling_probs)
  
        if s is not None:
            S[state-1, s] = S[state-1, s]+1

    return S
