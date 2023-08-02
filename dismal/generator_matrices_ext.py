import numpy as np
import math
import itertools


def q_diagonal(popsizes, migration_matrix):

    mig_rate_arr = migration_matrix[np.where(
        ~np.eye(migration_matrix.shape[0], dtype="bool"))]
    n_pop_combinations = math.comb(len(popsizes), 2)
    pops = range(len(popsizes))

    diagonal = ([-(1/popsizes[i] + np.sum(migration_matrix[i, :])) for i in range(len(popsizes))] +
                [-(np.sum(migration_matrix[pop1, :]) + np.sum(migration_matrix[pop2, :])
                   )/2 for (pop1, pop2) in itertools.combinations(pops, 2)]
                + [0])

    return diagonal


def deme_submatrix(migmat, deme_index):
    qm = np.diag([migmat[i, deme_index] for i in range(deme_index)])
    bottom = [migmat[deme_index, i] for i in range(deme_index)]

    qm = np.r_[qm, [bottom]]
    return qm


def qmig_submatrices(migmat, n_pops):

    submatrices = []

    for deme_idx in range(n_pops-1):
        m = deme_submatrix(migmat, deme_idx+1)
        submatrices.append(m)

    return submatrices


def pad_submatrices(submatrices):
    max_shape = submatrices[-1].shape

    padded = [np.pad(submat, ((0, max_shape[0]-submat.shape[0]),
                              (0, 0)),
                     'constant', constant_values=0) for submat in submatrices]

    return padded


def qmig(migmat):
    """Generate submatrix that describes migration rates in terms of a generator matrix"""
    n_pops = migmat.shape[0]
    submatrices = qmig_submatrices(migmat, n_pops)
    padded_submatrices = pad_submatrices(submatrices)
    return np.concatenate(pad_submatrices(submatrices), axis=1)


def qmig_star(migration_matrix):
    n_pops = migration_matrix.shape[0]
    combinations = list(itertools.combinations(range(n_pops), r=2))

    qmig_star = np.zeros(shape=(len(combinations), n_pops))

    for i in range(len(combinations)):
        pop1, pop2 = combinations[i]
        qmig_star[i, pop1] = migration_matrix[pop2, pop1]
        qmig_star[i, pop2] = migration_matrix[pop1, pop2]

    return qmig_star


def qmig_star_b(migration_matrix):
    n_pops = migration_matrix.shape[0]
    n = n_pops-1
    combinations = list(itertools.combinations(range(n_pops), r=2))

    qmig_star_b = np.zeros(shape=(len(combinations), len(combinations)))

    for i in range(len(combinations)):
        for j in range(len(combinations)):
            qmig_star_b[i, j] = migration_matrix[n-j, n-i]

    np.fill_diagonal(qmig_star_b, val=0)

    return qmig_star_b


def q(migration_matrix, pop_sizes):

    n_pops = len(pop_sizes)
    n_states = n_pops + math.comb(n_pops, 2) + 1
    q = np.zeros(shape=(n_states, n_states))

    q[0:n_pops, n_pops:-1] = qmig(migration_matrix)
    q[n_pops:-1, 0:n_pops] = qmig_star(migration_matrix)
    q[n_pops:-1, n_pops:-1] = qmig_star_b(migration_matrix)
    q[0:n_pops, -1] = np.array([1/popsize for popsize in pop_sizes])

    np.fill_diagonal(q, val=q_diagonal(
        migration_matrix=migration_matrix, popsizes=pop_sizes))

    return q
