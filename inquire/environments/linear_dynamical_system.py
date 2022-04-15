"""A submodule to define the Linear Dynamical System environment."""
from pathlib import Path

from inquire.environments.environment import Environment
from inquire.interactions.feedback import Trajectory

import numpy as np


class LinearDynamicalSystem(Environment):
    """A domain-agnostic linear dynamical system."""

    def __init__(
        self,
        seed: int = None,
        timesteps: int = 250,
        state_vector_size: int = 4,
        number_of_features: int = 6,
        trajectory_length: int = 10,
        control_space_discretization: int = 2000,
        output_path: str = str(Path.cwd())
        + "/output/linear_dynamical_system/",
        verbose: bool = True,
    ):
        """Initialize the Linear Dinamical System.

        We instantiate the generic LDS model:

            dx/dt = Ax(t) + Bu,

        with dynamics matrix A, state vector x, controls matrix B, and controls
        vector u. Assume A = B = I_{state_vector_size} to solve the simplified
        system:

            dx/dt = x(t) + u.

        In this system, features are <>

        ::inputs:
            ::seed: Seed used to generate the world for demonstrations and
                    preferences. Default 'None' assumes pseudo-random behavior.
            ::timesteps: The number of seconds it takes to execute a series of
                         controls.
            ::state_vector_size: Number of (arbitrary) variables that define a
                                 state.
            ::trajectory_length: The number of discrete states and controls
                                 in a trajectory. Note the term 'trajectory' is
                                 used for interpretability; 'sequence' might be
                                 a more apt term.
            ::control_space_discretization: How many discrete actions to
                                            generate from (-1,1) continuous
                                            control bounds.
        """
        super(LinearDynamicalSystem, self).__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._number_of_features = number_of_features
        self._trajectory_length = trajectory_length
        self._output_path = output_path
        self._state_vector_size = state_vector_size
        self._verbose = verbose

        # Initialize state with zeros:
        self._state = np.zeros(size=(state_vector_size, 1))
        # Randomly select goal state:
        self._goal_state = self._rng.integers(
            low=0, high=100, size=(self._state_vector_size, 1)
        )
        self._controls_bounds = np.array([[-1, 1]]).repeat(
            state_vector_size, axis=0
        )
        # Apply a control to each state-element for trajectory_duration time:
        self._controls_vector_full_trajectory = np.array(
            [self._trajectory_duration, self._state_vector_size]
        )
        self._controls_vector = np.full(
            (self._state_vector_size, control_space_discretization),
            fill_value=np.linespace(-1, 1, control_space_discretization),
        )

        try:
            assert timesteps % trajectory_length == 0
            self._controls_per_state = timesteps / trajectory_length
        except AssertionError:
            print(
                f"Timesteps ({timesteps}) isn't divisible by "
                f"trajectory length ({trajectory_length})."
            )
            return

    def generate_random_state(self, random_state):
        """Generate random start and goal states between (0, 1)."""
        self._state = random_state.random(size=self._state_vector_size)
        self._goal_state = random_state.random(size=self._state_vector_size)
        return

    def generate_random_reward(self, random_state):
        """Generate random configuration of weights between (0, 1)."""
        generated = random_state.random(size=self._state_vector_size)
        return generated

    def features(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Compute features from taking action when in state.

        ::inputs:
            ::action: The control to apply from state.
        """
        next_state = self.next_state(state, action)

        distance_from_goal_state = np.exp(
            -(np.abs(next_state - self._goal_state))
        )
        latest_features = np.empty((self.number_of_features, 1))

        # Extract features from next_state:
        for i, f in enumerate(self._number_of_features):
            latest_features[i, 0] = distance_from_goal_state ** i
        return latest_features

    def run(self, controls: np.ndarray) -> np.ndarray:
        """Generate trajectory from controls."""
        # Convert controls to account for time intervals:
        time_adjusted_controls = np.repeat(
            controls, self._controls_per_state, axis=0
        )
        trajectory = np.empty_like(time_adjusted_controls)
        trajectory[0, :] = self._state
        for i, t in enumerate(time_adjusted_controls):
            self._state = self._state + t
            trajectory[i, :] = self._state
            # See if we've reached the goal:
            if np.all(trajectory[i, :] == self._goal_state):
                # Reached the goal in fewer controls than provided; adjust
                # accordingly:
                trajectory = trajectory[: i + 1, :]
                time_adjusted_controls = time_adjusted_controls[: i + 1, :]
                break
        return [trajectory, time_adjusted_controls]

    def optimal_trajectory_from_w(self, start_state, w):
        """Compute the optimal trajectory to goal given weights w."""

        def reward(self, controls: np.ndarray) -> float:
            """Return reward for given controls and weights."""
            trajectory = self.run(controls)
            pass

        pass

    def all_actions(self):
        """Return continuous action-space."""
        return self._controls_vector

    def available_actions(self, current_state):
        """Return the actions available when in current_state."""
        if self.is_terminal_state(current_state):
            return None
        else:
            return self.all_actions

    def next_state(self, current_state, action):
        """Return effect on state if action taken in current_state."""
        if action.shape != self._controls_vector.shape:
            action_all_dimensions = np.full_like(self._controls_vector, action)
        next_state = self._state + action_all_dimensions
        return next_state

    def is_terminal_state(self, current_state):
        """Check if current_state signals execution completion."""
        if np.isclose(current_state, self._state):
            return True
        else:
            return False

    def state_space_dim(self):
        """Return dimensionality of observation space."""
        # The state-space is unbounded:
        return np.inf

    def state_space(self):
        """Return dimensionality of observation space."""
        return self.state_space_dim

    def state_index(self, state):
        """Observation-space is continuous; return None."""
        return None
