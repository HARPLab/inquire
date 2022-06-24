from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from inquire.environments.environment import Environment
from inquire.utils.datatypes import Query


class Agent(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_query(self, domain: Environment, query_state: np.ndarray, curr_w: np.ndarray) -> Query:
        pass

    @abstractmethod
    def update_weights(self, init_weights: np.ndarray, domain: Environment, feedback: list) -> Tuple[np.ndarray, np.ndarray]:
        # Returns tuple (w_dist, w_opt)
        pass
