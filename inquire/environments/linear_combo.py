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
        return np.zeros(self.w_dim)

    def generate_random_reward(self, random_state):
        generated = self._rng.normal(0, 1, size=(self.w_dim,))
        generated = generated / np.linalg.norm(generated)
        return generated

    def optimal_trajectory_from_w(self, start_state, w):
        return Trajectory([w], w)

    def features(self, action, state):
        return state

    def available_actions(self, current_state):
        sample = self._rng.normal(0, 1, (self.w_dim,))
        sample = sample / np.linalg.norm(sample,axis=0)
        return [sample]

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
