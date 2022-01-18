import pdb
from inquire.interactions.feedback import Trajectory
from inquire.utils.learning import Learning
import numpy as np
import math
import random
import time

class TrajectorySampling:

    @staticmethod
    def value_sampling(state, w_samples, domain, rand, steps, N, opt_params):
        """ Performs value iteraction according to weight samples (w_samples) and
        outputs trajectories containing actions that maximize value (if deterministic) or 
        actions based on a probability proportional to their value (if probabilistic).

        Parameters
        ----------
        state :
            The initial state that all sampled trajectories must start from
        w_samples : M x |w| numpy matrix
            The set of weight samples used to determine the value of state-action pairs
        domain : inquire.environments.Environment
            The domain to sample trajectories from
        rand : numpy.random.RandomState
            RandomState object, used to preserve deterministic sampling behavior across 
            multiple evaluations
        steps: int
            The maximum number of actions taken in each trajectory, which is met unless
            the agent reaches a terminal state
        N : int
            The number of trajectories to sample

        Additional Parameters (stored in dictionary "opt_params")
        ----------
        remove_duplicates : bool, optional
            Flag that sets whether all samples must have unique feature representations 
            (default is False)
        probabilistic : bool, optional
            Flag that sets whether actions should be selected deterministically (i.e., 
            always select the highest-value action) or probabilistically (i.e., choose
            an action with probability proportional to its value). Default is False.
        timeout : int, optional
            Maximum time to search for a new sample, measured in seconds. This timeout is
            considered only when searching for unique samples. Default is None.
        """

        samples, phis = [], []
        values = [Learning.discrete_q_iteration(domain, state, wi) for wi in w_samples]
        init_feats = domain.features(None,state)
        all_actions = domain.all_actions()
        last_addition = time.time()
        
        ## Parse optional arguments from dict
        remove_duplicates = opt_params.get("remove_duplicates", False)
        probabilistic = opt_params.get("probabilistic", False)
        timeout = opt_params.get("timeout", None)
        
        while len(samples) < N:
            vals = values[rand.randint(0,len(w_samples))]
            curr_state = state
            feats = [init_feats]
            traj = [[None,state]]
            for _ in range(steps):
                action_vals = np.exp(vals[tuple(domain.state_index(curr_state))])
                if probabilistic:
                    act_idx = np.random.choice(list(range(len(action_vals))), p=action_vals/np.sum(action_vals))
                else:
                    act_idx = np.argmax(action_vals)
                action = all_actions[act_idx]
                new_state = domain.next_state(curr_state, action)
                traj.append([action,new_state])
                feats.append(domain.features(action,new_state))
                curr_state = new_state
                if domain.is_terminal_state(curr_state):
                    break

            if remove_duplicates:
                phi = np.sum(feats,axis=0)
                dup = any([(phi == p).all() for p in phis])
                if any([(phi == p).all() for p in phis]):
                    if timeout is not None and (time.time() - last_addition > timeout):
                        return samples
                else:
                    samples.append(Trajectory(traj, np.sum(feats,axis=0)))
                    phis.append(phi)
                    last_addition = time.time()
            else:
                samples.append(Trajectory(traj, np.sum(feats,axis=0)))
        return samples

    @staticmethod
    def uniform_sampling(state, _, domain, rand, steps, N, opt_params):
        """ Samples N trajectories by randomly selecting an action for each step.

        Parameters
        ----------
        state :
            The initial state that all sampled trajectories must start from
        domain : inquire.environments.Environment
            The domain to sample trajectories from
        rand : numpy.random.RandomState
            RandomState object, used to preserve deterministic sampling behavior across 
            multiple evaluations
        steps: int
            The maximum number of actions taken in each trajectory, which is met unless
            the agent reaches a terminal state
        N : int
            The number of trajectories to sample

        Additional Parameters (stored in dictionary "opt_params")
        ----------
        remove_duplicates : bool, optional
            Flag that sets whether all samples must have unique feature representations 
            (default is False)
        timeout : int, optional
            Maximum time to search for a new sample, measured in seconds. This timeout is
            considered only when searching for unique samples.
        """

        samples, phis = [], []
        init_feats = domain.features(None,state)
        last_addition = time.time()

        ## Parse optional arguments from dict
        remove_duplicates = opt_params.get("remove_duplicates", False)
        timeout = opt_params.get("timeout", None)

        while len(samples) < N:
            curr_state = state
            traj = [[None, state]]
            feats = [init_feats]
            for _ in range(steps):
                actions = domain.available_actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if domain.is_terminal_state(curr_state):
                    break
            if remove_duplicates:
                phi = np.sum(feats,axis=0)
                dup = any([(phi == p).all() for p in phis])
                if any([(phi == p).all() for p in phis]):
                    if timeout is not None and (time.time() - last_addition > timeout):
                        return samples
                else:
                    samples.append(Trajectory(traj, np.sum(feats,axis=0)))
                    phis.append(phi)
                    last_addition = time.time()
            else:
                samples.append(Trajectory(traj, np.sum(feats,axis=0)))

        return samples

    @staticmethod
    def percentile_rejection_sampling(state, w_samples, domain, rand, steps, N, opt_params):
        """ Uniformly samples trajectories by randomly selecting an action for each step,
        then returns the N best trajectories according to the weight samples (w_samples).

        Parameters
        ----------
        state :
            The initial state that all sampled trajectories must start from
        w_samples : M x |w| numpy matrix
            The set of weight samples used to determine the value of state-action pairs
        domain : inquire.environments.Environment
            The domain to sample trajectories from
        rand : numpy.random.RandomState
            RandomState object, used to preserve deterministic sampling behavior across 
            multiple evaluations
        steps: int
            The maximum number of actions taken in each trajectory, which is met unless
            the agent reaches a terminal state
        N : int
            The number of trajectories to sample

        Additional Parameters (stored in dictionary "opt_params")
        ----------
        remove_duplicates : bool, optional
            Flag that sets whether all samples must have unique feature representations 
            (default is False)
        sample_size : float, optional
            Indicates how many trajectories should be initially sampled before selecting
            the top N samples (default is None; results in an exception) 
        timeout : int, optional
            Maximum time to search for a new sample, measured in seconds. This timeout is
            considered only when searching for unique samples.
        """

        sample_size = opt_params.get("sample_size", None)
        if sample_size is None:
            raise ValueError("sample_size is undefined")
        samples = TrajectorySampling.uniform_sampling(state, None, domain, rand, steps, sample_size, opt_params)
        rewards = np.stack([np.array([np.dot(t.phi, wi) for wi in w_samples]) for t in samples])
        var = np.var(rewards,axis=1)
        idxs = np.argsort(var)[::-1]
        top_samples = [samples[idxs[i]] for i in range(N)]
        return top_samples

