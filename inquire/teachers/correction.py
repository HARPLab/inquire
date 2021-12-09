import scipy.stats
from sampling import Sampling
import pdb
from feedback import Traj
import numpy as np
from itertools import combinations, permutations

class Correction:

    @staticmethod
    def sample_trajs(state, domain, w_samples, rand, N, steps):
        return Sampling.value_traj_sample(state, domain, w_samples, rand, N, steps)

    @staticmethod
    def get_prob_mat(exp):
        num_mat = np.broadcast_to(exp,(exp.shape[0],exp.shape[0],exp.shape[1]))
        trans_mat = np.transpose(num_mat,(1,0,2))
        den_mat = num_mat + trans_mat
        return trans_mat/den_mat

    @staticmethod
    def sum_over_choices(gains):
        return np.sum(gains,axis=0)

    @staticmethod
    def query_oracle(domain, trajs, query_idxs, display=False):
        ex_traj = trajs[0].traj
        start_state = ex_traj[0][1]
        curr_state = start_state
        feats = [domain.features(None,curr_state)]
        values = domain.q_iter_truth(start_state)
        traj = [[None,start_state]]
        comp_traj = trajs[query_idxs[0]]
        for step in range(len(ex_traj)-1):
            #actions = domain.actions(curr_state)
            #action_vals = []
            #for a in actions:
            #    new_state = domain.next_state(curr_state, a)
            #    v = values[new_state[0][0],new_state[0][1]]
            #    action_vals.append(v)
            #opt_action = actions[np.argmax(action_vals)]
            opt_action = domain.avail_actions[np.argmax(values[curr_state[0][0],curr_state[0][1]])]
            comp_state = comp_traj.traj[step+1]
            next_state = domain.next_state(curr_state, opt_action)
            next_action = opt_action
            if comp_traj.traj[step][1] == curr_state: 
                # Still following proposed trajectory
                comp_x, comp_y = comp_state[1][0]
                comp_val = np.max(values[comp_x,comp_y])
                ## Change to sample using Sampling.weighted_choice
                #opt_prob = (np.max(action_vals) - comp_val) / np.std(action_vals)

                max_prob = scipy.stats.norm.cdf(np.max(action_vals), loc=np.mean(action_vals), scale=np.std(action_vals))
                comp_prob = scipy.stats.norm.cdf(comp_val, loc=np.mean(action_vals), scale=np.std(action_vals))
                #if len(accepted_r) == 0 or np.random.rand() < np.mean(r)/(M*np.mean(accepted_r)):
                #if True: #len(accepted_r) == 0 or np.random.rand() > (np.mean(accepted_r) - np.mean(r))/np.mean(accepted_r):
                #if np.random.normal(0,1) >= opt_prob:
                if np.random.rand() >= max_prob - comp_prob:
                    next_state = comp_state[1]
                    next_action = comp_state[0]
            curr_state = next_state
            action = next_action
                
            traj.append([action,curr_state])
            feats.append(domain.features(action,curr_state))
        resp = Traj(traj, np.sum(feats,axis=0))
        if display:
            domain.display_traj(resp)
            domain.display_traj(comp_traj)
        return [resp,comp_traj]
