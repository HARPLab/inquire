from sampling import Sampling
import pdb
from feedback import Traj
import numpy as np
from itertools import combinations, permutations

class Demo:

    @staticmethod
    def sample_trajs(state, domain, w_samples, rand, N, steps):
        #return Sampling.uniform_traj_sample(state, domain, rand, N, steps)
        return Sampling.rejection_traj_sample(state, domain, w_samples, rand, N, steps)

    @staticmethod
    def get_sim_weighted_prob_mat(exp,sim):
        num = exp.T / np.sum(sim, axis=0)
        den = np.sum(num,axis=0)
        return (num/den).T

    @staticmethod
    def get_prob_mat(exp):
        prob_mat = exp / np.sum(exp,axis=0)
        return prob_mat

    @staticmethod
    def sum_over_choices(gains):
        return np.sum(gains)

    @staticmethod
    def query_oracle(domain, steps, trajs, query_idxs, display=False):
        ex_traj = trajs[0].traj
        start_state = ex_traj[0][1]
        curr_state = start_state
        feats = [domain.features(None,curr_state)]
        values = domain.q_iter_truth(start_state,flag=False)
        traj = [[None,start_state]]
        for step in range(steps):
            #actions = domain.actions(curr_state)
            #action_vals = []
            #for a in actions:
            #    new_state = domain.next_state(curr_state, a)
            #    v = values[new_state[0][0],new_state[0][1]]
            #    action_vals.append(v)
            #action = actions[np.argmax(action_vals)]
            opt_action = domain.avail_actions[np.argmax(values[curr_state[0][0],curr_state[0][1]])]
            curr_state = domain.next_state(curr_state, opt_action)
            traj.append([opt_action,curr_state])
            feats.append(domain.features(opt_action,curr_state))
            if curr_state[0] == curr_state[1]: # Check if terminal state
                break
        resp = Traj(traj, np.sum(feats,axis=0))
        if display:
            domain.display_traj(resp)
        return [resp]+trajs
