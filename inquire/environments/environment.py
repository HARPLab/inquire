from abc import ABC, abstractmethod
from inquire.utils.datatypes import CachedSamples
import math
import numpy as np


class CachedTask:
    def __init__(self, state_samples: CachedSamples, num_query_states, num_test_states):
        task = state_samples[0].task
        for s in state_samples:
            assert (s.task._r == task._r).all()
        self._r = task._r
        self.domain = task.domain
        self.query_states = state_samples[:num_query_states]
        self.test_states = state_samples[num_query_states:num_query_states+num_test_states]

    def get_ground_truth(self):
        return self._r

    def optimal_trajectory_from_ground_truth(self, start_state):
        return start_state.best_traj

    def least_optimal_trajectory_from_ground_truth(self, start_state):
        return start_state.worst_traj

    def ground_truth_reward(self, trajectory):
        return np.dot(self._r, trajectory.phi)

    def distance_from_ground_truth(self, w):
        cos = np.dot(self._r, w) / (np.linalg.norm(self._r) * np.linalg.norm(w))
        return np.arccos(cos) / math.pi

class Task:
    def __init__(self, domain, num_query_states, num_test_states, random_state):
        self.rand = random_state
        self.domain = domain
        self.test_states, self.query_states = [], []
        for _ in range(num_query_states):
            self.query_states.append(self.domain.generate_random_state(self.rand))
        for _ in range(num_test_states):
            self.test_states.append(self.domain.generate_random_state(self.rand))
        self._r = self.domain.generate_random_reward(self.rand)

    def get_ground_truth(self):
        return self._r

    def optimal_trajectory_from_ground_truth(self, start_state):
        return self.domain.optimal_trajectory_from_w(start_state, self._r)

    def least_optimal_trajectory_from_ground_truth(self, start_state):
        return self.domain.optimal_trajectory_from_w(start_state, -1 * self._r)

    def ground_truth_reward(self, trajectory):
        return np.dot(self._r, trajectory.phi)

    def distance_from_ground_truth(self, w):
        cos = np.dot(self._r, w) / (np.linalg.norm(self._r) * np.linalg.norm(w))
        return np.arccos(cos) / math.pi

class Environment(ABC):
    @abstractmethod
    def w_dim(self):
        pass

    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def generate_random_state(self, random_state):
        pass

    @abstractmethod
    def generate_random_reward(self, random_state):
        pass

    @abstractmethod
    def optimal_trajectory_from_w(self, start_state, w):
        pass

    @abstractmethod
    def trajectory_rollout(self, start_state, actions):
        pass

    @abstractmethod
    def features_from_trajectory(self, trajectory):
        pass

    @abstractmethod
    def distance_between_trajectories(self, a, b):
        pass

    @abstractmethod
    def visualize_trajectory(self, start_state, trajectory):
        pass

