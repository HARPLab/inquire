"""A submodule to define the Linear Dynamical System environment."""
from pathlib import Path

from inquire.environments.environment import Environment

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
        output_path: str = str(Path.cwd())
        + "/output/linear_dynamical_system/",
        verbose: bool = True,
    ):
        """Initialize the Linear Dinamical System.

        We instantiate the generic LDS model:

            dx/dt = Ax(t) + Bu,

        with dynamics matrix A, state vector x, controls matrix B, and controls
        vector u. We assume A = B = I_{state_vector_size}.

        ::inputs:
            ::seed: Seed used to generate the world for demonstrations and
                    preferences.
            ::timesteps: The number of seconds it takes to execute a series of
                         controls.
            ::state_vector_size: Number of (arbitrary) variables that define a
                                 state.
            ::trajectory_length: The number of discrete states and controls
                                 in a trajectory.
        """
        super(LinearDynamicalSystem, self).__init__()
        self._verbose = verbose
        self._trajectory_length = trajectory_length
        self._output_path = output_path
        self._state_vector_size = state_vector_size
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(size=(state_vector_size, 1))
        self._controls_vector = np.zeros(size=(state_vector_size, 1))

        # Randomly select goal state:
        self._goal_state = self._rng.integers(
            low=0, high=100, size=(number_of_features, 1)
        )

        # Dynamics matrix should have fixed values:
        self._dynamics_matrix = np.ones(
            (state_vector_size, state_vector_size), dtype=np.float32
        )
        # Controls matrix should have fixed values:
        self._controls_matrix = np.ones(
            (state_vector_size, state_vector_size), dtype=np.float32
        )

        self._controls_bounds = np.array([[-1, 1]]).repeat(
            state_vector_size, axis=0
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
        self._state = random_state.random(size=self._state.shape)
        self._goal_state = random_state.random(size=self._state.shape)
        return

    def generate_random_reward(self, random_state):
        """Generate random configuration of weights between (0, 1)."""
        generated = random_state.random(size=self._state.shape)
        return generated

    def features(self, action, state):
        """Compute features from taking action when in state."""
        next_state = self.next_state(state, action)
        pass

    def optimal_trajectory_from_w(self, start_state, w):
        pass

    def all_actions(self):
        """Return continuous action-space."""
        actions = np.linspace(-1, 1, 10000)
        return actions

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
        pass

    def state_space(self):
        pass

    def state_index(self, state):
        """Return None since observation-space is continuous."""
        return None
