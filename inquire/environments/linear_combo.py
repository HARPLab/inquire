from inquire.environments.environment import Environment
from inquire.utils.datatypes import Trajectory, Range
import numpy as np
import math


class LinearCombination(Environment):
    def __init__(self, seed, w_dim):
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self.feat_dim = w_dim
        self.action_rang = Range(
            np.array([-1]*w_dim),
            np.ones((w_dim)),
            np.array([1]*w_dim),
            np.ones((w_dim))
        )
        self.state_rang = self.action_rang
        self.trajectory_length = 1

    def w_dim(self):
        return self.feat_dim

    def action_space(self):
        return self.action_rang

    def state_space(self):
        return self.state_rang

    def generate_random_state(self, random_state):
        return np.zeros(self.feat_dim)

    def generate_random_reward(self, random_state):
        generated = self._rng.normal(0, 1, size=(self.feat_dim,))
        generated = generated / np.linalg.norm(generated)
        return generated

    def optimal_trajectory_from_w(self, start_state, w):
        return Trajectory(states=np.expand_dims(w,axis=0), actions=None, phi=w)

    def trajectory_rollout(self, start_state, actions):
        action = actions[-self.feat_dim:]
        state = action/np.linalg.norm(action)
        return Trajectory(states=np.expand_dims(state,axis=0), actions=None, phi=state)

    def features_from_trajectory(self, trajectory):
        return trajectory.states[-1]

    def distance_between_trajectories(self, a, b):
        cos = np.dot(a.phi, b.phi) / (np.linalg.norm(a.phi) * np.linalg.norm(b.phi))
        return np.arccos(np.clip(cos,-1,1)) / math.pi

    def visualize_trajectory(self, start_state, trajectory):
        print(trajectory.states[-1])
