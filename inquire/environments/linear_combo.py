import math
from inquire.environments.environment import Environment
from inquire.interactions.feedback import Trajectory
import numpy as np
import pdb

class LinearCombination(Environment):
    def __init__(self, seed, w_dim):
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self.w_dim = w_dim

    def generate_random_state(self, random_state):
        self.state_len = self._rng.uniform() * np.sqrt(self.w_dim)
        return np.zeros(self.w_dim)

    def generate_random_reward(self, random_state):
        generated = self._rng.uniform(low=-1, high=1, size=(self.w_dim,))
        generated = generated / np.linalg.norm(generated)
        return generated

    def optimal_trajectory_from_w(self, start_state, w):
        sorted_idxs = np.argsort(w)[::-1]
        state = np.zeros_like(w)
        for i in range(w.shape[0]):
            diff = math.sqrt(max(0,self.state_len**2.0 - np.linalg.norm(state)**2.0))
            state[sorted_idxs[i]] = min(diff, 1.0)
        return Trajectory([state], state)

    def features(self, action, state):
        return state

    def available_actions(self, current_state):
        sample = self._rng.uniform(0, 1, (self.w_dim,))
        sample = self.state_len * sample / np.linalg.norm(sample,axis=0)
        return [np.clip(sample,0,1)]

    def next_state(self, current_state, action):
        return action

    def is_terminal_state(self, current_state):
        return np.any(current_state != 0)
    
    def all_actions(self):
        return None

    def state_space_dim(self):
        return None

    def state_space(self):
        return None

    def state_index(self, state):
        return None

    def trajectory_from_states(self, states, features):
        return Trajectory(states, np.sum(features, axis=0))

    def distance_between_trajectories(self, a, b):
        return np.linalg.norm(a.trajectory[-1][1] - b.trajectory[-1][1])
