from inquire.interactions.feedback import Trajectory
from inquire.utils.learning import Learning
import numpy as np
import math
import random

class TrajectorySampling:

    @staticmethod
    def value_sampling(state, domain, w_samples, rand, N, steps, probabilistic=False, remove_duplicates=False):
        trajs = []
        values = [Learning.discrete_q_iteration(domain, state, wi) for wi in w_samples]
        init_feats = domain.features(None,state)
        
        for i in range(N):
            vals = values[rand.randint(0,len(w_samples))]
            curr_state = state
            feats = [init_feats]
            traj = [[None,state]]
            for step in range(steps):
                actions = domain.all_actions()
                action_vals = np.exp(vals[domain.state_index(curr_state)])
                if probabilistic:
                    choice_weights = action_vals/np.sum(action_vals)
                    act_idx = np.random.choice(list(range(len(action_vals))), p=choice_weights)
                else:
                    act_idx = np.argmax(action_vals)
                action = actions[act_idx]
                curr_state = domain.next_state(curr_state, action)
                traj.append([action,curr_state])
                feats.append(domain.features(action,curr_state))
                if domain.is_terminal_state(curr_state):
                    break

            if remove_duplicates:
                duplicate = False
                new_traj = Trajectory(traj, np.sum(feats,axis=0))
                for t in trajs:
                    if (new_traj.phi == t.phi).all():
                        duplicate = True
                        break
                if not duplicate:
                    trajs.append(new_traj)
            else:
                trajs.append(Trajectory(traj, np.sum(feats,axis=0)))
        return trajs

    @staticmethod
    def uniform_sampling(state, _, domain, rand, steps, N):
        samples = []
        for i in range(N):
            curr_state = state
            traj = [[None, curr_state]]
            feats = [domain.features(None,curr_state)]
            for j in range(steps):
                actions = domain.available_actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if domain.is_terminal_state(curr_state):
                    break
            samples.append(Trajectory(traj, np.sum(feats,axis=0)))
        return samples

    @staticmethod
    def percentile_rejection_sampling(state, domain, w_samples, rand, N, steps, pct=0.8):
        sample_size = math.ceil(float(N) / (1.0-pct))
        samples, rewards = [], []
        for _ in range(sample_size):
            curr_state = state
            traj = [[None, curr_state]]
            feats = [domain.features(None,curr_state)]
            for j in range(steps):
                actions = domain.actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if curr_state[0] == curr_state[1]: # Check if terminal state
                    break

                phi = np.sum(feats,axis=0)
                new_traj = Trajectory(traj, phi)
                samples.append(new_traj)
                r = [np.dot(new_traj.phi,wi) for wi in w_samples]
                rewards.append(r)
        var = np.var(rewards,axis=1)
        mean = np.mean(rewards,axis=1)
        rang = np.ptp(rewards,axis=1)
        idxs = np.argsort(var)[::-1]
        top_samples = [samples[idxs[i]] for i in range(N)]
        return top_samples

    @staticmethod
    def rejection_sampling(state, domain, w_samples, rand, N, steps, threshold=0.1, burn=1000):
        samples, accepted_r = [], []
        accepted_flag, accepted_all, accepted_full = [],[], []
        while len(accepted_flag) < (burn + N):
            curr_state = state
            traj = [[None, curr_state]]
            feats = [domain.features(None,curr_state)]
            for j in range(steps):
                actions = domain.actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if curr_state[0] == curr_state[1]: # Check if terminal state
                    break

                phi = np.sum(feats,axis=0)
                new_traj = Trajectory(traj, phi)
                r = [np.dot(new_traj.phi,wi) for wi in w_samples]

                if len(accepted_r) < 2 or np.std(accepted_r) == 0:
                    opt_prob = np.inf 
                else:
                    #try max instead of np.mean(r) next
                    opt_prob = scipy.stats.norm.cdf(np.mean(r), loc=np.mean(accepted_r), scale=np.std(accepted_r))
                if threshold < 0:
                    flag = np.random.rand() < opt_prob
                else:
                    flag = opt_prob > 0 and np.log(np.random.rand()) < np.log(opt_prob) - np.log(threshold)
                if flag:
                    samples.append(new_traj)
                    accepted_r.append(np.mean(r))
                    accepted_all.append(r)
            if flag:
                accepted_flag.append(r)
            accepted_full.append(r)
        metrics_all = [np.mean(np.median(accepted_all,axis=1)), np.mean(np.var(accepted_all,axis=1)), np.mean(np.ptp(accepted_all,axis=1))]
        metrics_full = [np.mean(np.median(accepted_full,axis=1)), np.mean(np.var(accepted_full,axis=1)), np.mean(np.ptp(accepted_full,axis=1))]
        if len(accepted_flag) > 0:
            metrics_flag = [np.mean(np.median(accepted_flag,axis=1)), np.mean(np.var(accepted_flag,axis=1)), np.mean(np.ptp(accepted_flag,axis=1))]
        pdb.set_trace()
        return samples[burn:]
