from sampling import Sampling
import pdb
import numpy as np
from itertools import combinations, permutations

class Pref:

    @staticmethod
    def sample_trajs(state, domain, w_samples, rand, N, steps):
        return Sampling.uniform_traj_sample(state, domain, rand, N, steps)

    @staticmethod
    def get_prob_mat(exp):
        num_mat = np.broadcast_to(exp,(exp.shape[0],exp.shape[0],exp.shape[1]))
        trans_mat = np.transpose(num_mat,(1,0,2))
        den_mat = num_mat + trans_mat 
        return trans_mat/den_mat

    @staticmethod
    def sum_over_choices(gains):
        return gains + gains.T

    @staticmethod
    def query_oracle(domain, steps, trajs, query_idxs, display=False):
        q = [trajs[idx] for idx in query_idxs]
        r = [domain.trace_reward_truth(qi.phi) for qi in q]
        ordering = [qi for _, qi in sorted(zip(r, q))][::-1]
        if display:
            domain.display_traj(ordering[0])
            domain.display_traj(ordering[1])
        return ordering #ordering
