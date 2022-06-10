from typing import Union
from enum import Enum
import numpy as np

class Modality(Enum):
    DEMONSTRATION = 0
    PREFERENCE = 1
    CORRECTION = 2
    BINARY = 3
    DEMONSTRATION_PAIRWISE = 4 
    NONE = -1

class CachedSamples:
    def __init__(self, task, state, best_traj, worst_traj, traj_samples):
        self.task = task
        self.state = state
        self.best_traj = best_traj
        self.worst_traj = worst_traj
        self.traj_samples = traj_samples

class Query:
    def __init__(
            self,
            query_type: Modality,
            start_state: Union[list, np.ndarray],
            trajectories: list
    ):
        self.query_type = query_type
        self.start_state = start_state
        self.trajectories = trajectories

class Trajectory:
    def __init__(self, states: list, actions: list, phi: Union[list, np.ndarray]):
        self.phi = phi
        self.states = states
        self.actions = actions

class Choice:
    def __init__(self, selection, options):
        self.selection = selection
        self.options = options

class Feedback:
    def __init__(self, modality, query, choice):
        self.modality = modality
        self.query = query
        self.choice = choice

class Range:
    def __init__(self, min_vals, min_inclusive, max_vals, max_inclusive):
        assert min_vals.shape[0] == max_vals.shape[0]
        self.dim = min_vals.shape[0]
        self.min = min_vals
        self.min_inclusive = min_inclusive
        self.max = max_vals
        self.max_inclusive = max_inclusive
