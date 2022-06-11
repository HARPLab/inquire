import pdb
from inquire.utils.datatypes import Range, Trajectory, CachedSamples
import numpy as np
import math
import random
import time

class TrajectorySampling:

    @staticmethod
    def uniform_sampling(state, _, domain, rand, steps, N, opt_params):
        if isinstance(state, CachedSamples):
            return rand.choice(state.traj_samples, N)

        action_samples = []
        action_space = domain.action_space()
        if isinstance(action_space, Range):
            action_samples = np.full((N,steps,action_space.dim), np.inf)
            for i in range(action_space.dim):
                while (action_samples[:,:,i] == np.inf).any():
                    ai = rand.uniform(low=action_space.min[i], high=action_space.max[i], size=(N,steps))
                    within_min = action_space.min_inclusive[i] or (ai > action_space.min[i]).all()
                    within_max = action_space.max_inclusive[i] or (ai < action_space.max[i]).all()
                    if within_min and within_max:
                        action_samples[:,:,i] = ai
        else:
            action_samples = np.stack([rand.choice(action_space[i],size=(N,steps)) for i in range(action_space.shape[0])],axis=-1)

        trajectories = [domain.trajectory_rollout(state, action_samples[i].flatten()) for i in range(N)]
        return trajectories
