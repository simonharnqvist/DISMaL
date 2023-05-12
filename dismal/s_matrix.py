import numpy as np
import random

class SMatrix:

    def __init__(self):
        pass
        

    @staticmethod
    def s_between_two_arrs(arr1, arr2):
        assert len(arr1) == len(arr2), "Haplotype blocks must be of same length"    
        return len(arr1) - np.count_nonzero(arr1 == arr2)
    
    @staticmethod
    def counts_to_dict(x):
        """
        Generate dictionary of counts of nucleotide difference (s-value) occurrences.
        :param list x: list of nt differences between loci
        :return: s_dict
        """
        s, s_count = np.unique(x, return_counts=True)
        return dict(zip(s,s_count))
    
    @staticmethod
    def s_matrix_from_dicts(dicts):
        s_max = int(np.max([np.max(list(dicts[i].keys())) for i in [0,1,2]]))
        s_matrix = []

        for i in [0,1,2]:
            s_vals = [dicts[i].get(j,0) for j in range(0, s_max)] # return 0 if key not found
            s_matrix.append(s_vals)

        return np.array(s_matrix)

    def generate_from_blockdict(self, blockdict, samples_dict):

        s1 = []
        s2 = []
        s3 = []

        populations = list(set(samples_dict.values()))

        for chr in blockdict.keys():
            for block in blockdict[chr].keys():
                samples = random.sample([sample for sample in blockdict[chr][block].keys()], 2)
                haplotypes = random.choices(["hap_1", "hap_2"], k=2) # with replacement
                sampled_haps = [blockdict[chr][block][samples[i]][haplotypes[i]] for i in [0,1]]
                s = self.s_between_two_arrs(sampled_haps[0], sampled_haps[1])

                pop1 = samples_dict.get(samples[0])
                pop2 = samples_dict.get(samples[1])
                assert pop1 in populations, print(f"{pop1} not in populations {populations}")
                assert pop2 in populations, print(f"{pop2} not in populations {populations}")

                if pop1 != pop2:
                    s3.append(s)
                elif pop1 == pop2 == populations[0]:
                    s1.append(s)
                else: 
                    assert pop1 == pop2 == populations[1]
                    s2.append(s)

        s_dicts = [self.counts_to_dict(s) for s in [s1, s2, s3]]

        return self.s_matrix_from_dicts(s_dicts)