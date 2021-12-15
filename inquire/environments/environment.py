from abc import ABC, abstractmethod
import numpy as np

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

    def ground_truth_reward(self, trajectory):
        return np.dot(self._r, trajectory.phi)

    def distance_from_ground_truth(self, w):
        return np.linalg.norm(self._r - w)

class Environment(ABC):
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
    def features(self, action, state):
        pass

    @abstractmethod
    def available_actions(self, current_state):
        pass

    @abstractmethod
    def next_state(self, current_state, action):
        pass

    @abstractmethod
    def is_terminal_state(self, current_state):
        pass

    @abstractmethod
    def all_actions(self):
        pass

    @abstractmethod
    def state_space_dim(self):
        pass

    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def state_index(self, state):
        pass
