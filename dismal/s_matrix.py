import numpy as np
import random
from dismal.preprocess import read_r_simulation


class SMatrix:

    def __init__(self):
        pass

    def from_rgim_simulation(self, paths):
        assert len(paths) == 3

        X = [read_r_simulation(path) for path in paths]
        S = self.from_lists(X)
        return S

    def from_lists(self, s_lists):
        """Generate S-matrix from lists of counts"""
        max_len = np.max([np.max(l) for l in s_lists])
        s_matrix = np.zeros(shape=(3, max_len+1))

        for i in [0, 1, 2]:  # for each state
            idx, count = np.unique(s_lists[i], return_counts=True)
            for j in range(0, len(idx)):
                s_matrix[i][idx[j]] = count[j]

        return np.array(s_matrix)

    @staticmethod
    def s_between_two_arrs(arr1, arr2, ignore_missing=True):
        assert len(arr1) == len(
            arr2), "Haplotype blocks must be of same length"

        if ignore_missing:
            # here ignore = remove element from both arrays if missing in either; missingness encoded with negative values
            arr1_missing = np.where(arr1 < 0)[0]
            arr2_missing = np.where(arr2 < 0)[0]
            missing_idx = np.unique(
                np.concatenate([arr1_missing, arr2_missing]))
            arr1 = np.delete(arr1, missing_idx)
            arr2 = np.delete(arr2, missing_idx)

        return len(arr1) - np.count_nonzero(arr1 == arr2)

    @staticmethod
    def counts_to_dict(x):
        """
        Generate dictionary of counts of nucleotide difference (s-value) occurrences.
        :param list x: list of nt differences between loci
        :return: s_dict
        """
        s, s_count = np.unique(x, return_counts=True)
        return dict(zip(s, s_count))

    def from_dicts(self, dicts):
        s_max = int(np.max([np.max(list(dicts[i].keys())) for i in [0, 1, 2]]))
        s_matrix = []

        for i in [0, 1, 2]:
            s_vals = [dicts[i].get(j, 0) for j in range(
                0, s_max)]  # return 0 if key not found
            s_matrix.append(s_vals)

        return np.array(s_matrix)

    def from_blockdict(self, blockdict, samples_dict, ignore_missing=True):

        s1 = []
        s2 = []
        s3 = []

        populations = list(set(samples_dict.values()))

        for chrom in blockdict.keys():
            for block in blockdict[chrom].keys():
                samples = list(blockdict[chrom][block].keys())

                if len(samples) >= 2:
                    chosen_samples = random.sample(samples, k=2)
                    haplotypes = random.choices(
                        ["hap_1", "hap_2"], k=2)  # with replacement
                    sampled_haps = [blockdict[chrom][block]
                                    [chosen_samples[i]][haplotypes[i]] for i in [0, 1]]
                    s = self.s_between_two_arrs(sampled_haps[0], sampled_haps[1],
                                                ignore_missing=ignore_missing)

                    pop1 = samples_dict.get(chosen_samples[0])
                    pop2 = samples_dict.get(chosen_samples[1])
                    assert pop1 in populations, print(
                        f"{pop1} not in populations {populations}")
                    assert pop2 in populations, print(
                        f"{pop2} not in populations {populations}")

                    if pop1 != pop2:
                        s3.append(s)
                    elif pop1 == pop2 == populations[0]:
                        s1.append(s)
                    else:
                        assert pop1 == pop2 == populations[1]
                        s2.append(s)

        s_dicts = [self.counts_to_dict(s) for s in [s1, s2, s3]]

        return self.from_dicts(s_dicts)
