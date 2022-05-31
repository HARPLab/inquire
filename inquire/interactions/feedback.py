from typing import Union
from enum import Enum
import numpy as np

class Modality(Enum):
    DEMONSTRATION = 0
    PREFERENCE = 1
    CORRECTION = 2
    BINARY = 3
    NONE = -1

class Query:
    def __init__(
            self,
            query_type: Modality,
            task: object,
            start_state: Union[list, np.ndarray],
            trajectories: list
    ):
        self.query_type = query_type
        self.task = task  # A task has an Environment as an instance attribute
        self.start_state = start_state
        self.trajectories = trajectories


class Trajectory:
    def __init__(self, trajectory: list, phi: Union[list, np.ndarray]):
        self.trajectory = trajectory
        self.phi = phi

class Choice:
    def __init__(self, selection, options):
        self.selection = selection
        self.options = options

class Feedback:
    def __init__(self, modality, choice):
        self.modality = modality
        self.choice = choice
