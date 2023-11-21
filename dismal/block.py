import numpy as np
import itertools

class Block:
    
    def __init__(self, callset, start_idx, blocklen, pop1_samples, pop2_samples):
        self.start_idx = start_idx
        self.end_idx = start_idx + blocklen
        self.blocklen = blocklen
        self.samples = callset.samples
        self.callset = callset
        self.pop1_samples = pop1_samples
        self.pop2_samples = pop2_samples

        self.gt = self.make_block()
        self.pop1_gt, self.pop2_gt = self.split_by_pop()
        self.pop1_gt_flat = np.array([self.pop1_gt[i].flatten() for i in range(len(self.pop1_gt))])
        self.pop2_gt_flat = np.array([self.pop2_gt[i].flatten() for i in range(len(self.pop2_gt))])

        self.pop1_gt_filtered = self.remove_missing_haps(self.pop1_gt_flat)
        self.pop2_gt_filtered = self.remove_missing_haps(self.pop2_gt_flat)

        self.s1 = self.within_state_s(self.pop1_gt_filtered)
        self.s2 = self.within_state_s(self.pop2_gt_filtered)
        self.s3 = self.between_states_s(self.pop1_gt_filtered, self.pop2_gt_filtered)


    def make_block(self):
        """Obtain indices in callset corresponding to block"""
        return self.callset.gt[np.where((self.callset.pos >= self.start_idx) 
                & (self.callset.pos < (self.start_idx+self.blocklen)))]


    def split_by_pop(self):
        """Split GT calls by population"""

        pop1_idxs = [list(self.samples).index(sampl_id) for sampl_id in self.pop1_samples]
        pop2_idxs = [list(self.samples).index(sampl_id) for sampl_id in self.pop2_samples]

        return self.gt[:, pop1_idxs], self.gt[:, pop2_idxs]
    

    @staticmethod
    def remove_missing_haps(arr):
        return arr[:, ~(arr == -1).any(axis=0)]


    def within_state_s(self, block_gt):
        """Calculate the number of segregating sites between each haplotype from a single population within a block"""

        if block_gt is not None:
            s = [np.sum(a != b) for (a,b) in list(itertools.combinations([block_gt[:, hapl] for hapl in range(block_gt.shape[1])], 2))]
        else:
            s = None

        return s
    
    
    def between_states_s(self, pop1_block_gt, pop2_block_gt):
        """Calculate the number of segregating sites between each haplotype from different populations within a block"""

        if pop1_block_gt is not None and pop2_block_gt is not None:
            s = [np.sum(a != b) for (a,b) in list(itertools.product([pop1_block_gt[:, hapl] 
                                    for hapl in range(pop1_block_gt.shape[1])], 
                                    [pop2_block_gt[:, hapl] 
                                    for hapl in range(pop2_block_gt.shape[1])]))]
        else:
            s = None

        return s
