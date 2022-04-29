"""A submodule to define the Linear Dynamical System environment."""
import sys
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as opt
from inquire.environments.environment import Environment
from inquire.interactions.feedback import Trajectory


class LinearDynamicalSystem(Environment):
    """A domain-agnostic linear dynamical system."""

    def __init__(
        self,
        seed: int = None,
        timesteps: int = 150,
        state_vector_size: int = 4,
        number_of_features: int = 8,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: np.ndarray = 100,
        control_space_discretization: int = 2000,
        output_path: str = str(Path.cwd())
        + "/output/linear_dynamical_system/",
        verbose: bool = True,
    ):
        """Initialize the Linear Dinamical System.

        We instantiate the generic LDS model:

            dx/dt = Ax(t) + Bu(t),

        with dynamics matrix A, state vector x, controls matrix B, and controls
        vector u. Assume A = B = I_{state_vector_size} to solve the simplified
        system:

            dx/dt = x(t) + u(t).

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
                                 a more apt term in this context.
            ::control_space_discretization: How many discrete actions to
                                            generate from (-1,1) continuous
                                            control bounds.
        """
        super(LinearDynamicalSystem, self).__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.w_dim = number_of_features
        self._trajectory_length = trajectory_length
        self._output_path = output_path
        self._state_vector_size = state_vector_size
        self._optimal_trajectory_iterations = optimal_trajectory_iterations
        self._verbose = verbose

        # Randomly select goal state:
        self._goal_state = self._rng.integers(
            low=0,
            high=10 * self._trajectory_length,
            size=(self._state_vector_size, 1),
        )
        self._controls_bounds = np.array([[-1, 1]]).repeat(
            state_vector_size, axis=0
        )
        self._optimizers_controls_bounds = self._controls_bounds.repeat(
            self._trajectory_length, axis=0
        )
        self._lower_bound = [
            x[0] for x in self._controls_bounds
        ] * self._trajectory_length
        self._upper_bound = [
            x[1] for x in self._controls_bounds
        ] * self._trajectory_length
        # Apply a control to each state-element for timesteps time:
        self._controls_vector = np.empty((self._state_vector_size,))
        self._controls_matrix = np.full(
            (self._state_vector_size, control_space_discretization),
            fill_value=np.linspace(-1, 1, control_space_discretization),
        )

        try:
            assert timesteps % trajectory_length == 0
            self._controls_per_state = int(timesteps / trajectory_length)
        except AssertionError:
            print(
                f"Timesteps ({timesteps}) isn't divisible by "
                f"trajectory length ({trajectory_length})."
            )
            return

    def reset(self, seed: int = None) -> None:
        """Reset the environment according to seed.

        Note: Start state is always the 0-vector.
        """
        if seed is None:
            self._rng = np.random.default_rng(self._seed)
        else:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.random((self._state_vector_size, 1))
        self._goal_state = self._rng.integers(
            low=0,
            high=10 * self._trajectory_length,
            size=(self._state_vector_size, 1),
        )

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate random goal as state vector."""
        rando = random_state.randint(low=0, high=sys.maxsize)
        return rando

    def generate_random_reward(self, random_state) -> np.ndarray:
        """Generate weights of random value between (0, 1)."""
        generated = self._rng.random(size=(self.w_dim,))
        generated = generated / generated.sum()
        return generated

    def features(
        self, action: np.ndarray, state: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Compute features of state.

        ::inputs:
            ::state: The CURRENT state.
            ::action: The control to apply from state.
        """
        if type(state) is int:
            self._seed = state
            self._state = self._rng.random((self._state_vector_size, 1))
            state = self._state
        if action is None:
            action = np.zeros_like(state)
        action = np.array(action).reshape(-1, 1)
        state = state.reshape(-1, 1)
        s_prime = (state + action).reshape(-1, 1)
        s_diff = np.exp(-np.abs(s_prime - self._goal_state))
        latest_features = np.concatenate((s_diff, -action.reshape(-1, 1)))
        return latest_features.squeeze()

    def run(self, controls: np.ndarray) -> np.ndarray:
        """Generate trajectory from controls."""
        controls = controls.reshape(-1, self._controls_vector.shape[0])
        trajectory = np.empty(
            (
                int(self._trajectory_length * self._controls_per_state),
                self._state_vector_size,
            )
        )
        time_adjusted_controls = controls.repeat(
            self._controls_per_state, axis=0
        )
        self.reset()
        for i, u in enumerate(time_adjusted_controls):
            trajectory[i, :] = self._state.squeeze()
            if self.is_terminal_state(trajectory[i, :]):
                # Reached goal in fewer controls than provided; adjust
                # accordingly:
                trajectory = trajectory[: i + 1, :]
                time_adjusted_controls = time_adjusted_controls[: i + 1, :]
                return [trajectory, time_adjusted_controls]
            self._state = self._state + u.reshape(-1, 1)
        return [trajectory, time_adjusted_controls]

    def optimal_trajectory_from_w(self, start_state, w):
        """Compute the optimal trajectory to goal given weights w.

        In this formulation, the smaller the feature-values, the higher the
        reward.

        ::inputs:
            ::start_state: A seed with which we reset the environment.
            ::w: A set of weights.
        """
        self._seed = start_state
        self.reset(start_state)

        def reward_fn(
            controls: np.ndarray, domain: Environment, weights: np.ndarray
        ) -> float:
            """Return reward for given controls and weights."""
            trajectory = self.run(controls)
            features = np.zeros((self.w_dim,))
            for i in range(trajectory[0].shape[0]):
                features += self.features(
                    trajectory[1][i, :], trajectory[0][i, :]
                )
            features = features / trajectory[0].shape[0]
            rwd = features.T @ w
            return -(rwd.squeeze())

        optimal_ctrl = None
        opt_val = np.inf
        start = time.perf_counter()
        # Find the optimal controls given start_state and weights w:
        for _ in range(self._optimal_trajectory_iterations):
            if self._verbose:
                print(
                    f"Beginning optimization iteration {_+1} of "
                    f"{self._optimal_trajectory_iterations}."
                )
            temp_result = opt.fmin_l_bfgs_b(
                reward_fn,
                x0=np.random.uniform(
                    low=self._lower_bound,
                    high=self._upper_bound,
                    size=self._controls_vector.shape[0]
                    * self._trajectory_length,
                ),
                args=(self, w),
                bounds=self._optimizers_controls_bounds,
                approx_grad=True,
                # maxfun=1000,
                # maxiter=100,
            )
            if temp_result[1] < opt_val:
                optimal_ctrl = temp_result[0]
                opt_val = temp_result[1]
        elapsed = time.perf_counter() - start
        if self._verbose:
            print(
                "Finished generating optimal trajectory in "
                f"{elapsed:.4f} seconds."
            )

        optimal_trajectory = self.run(optimal_ctrl)
        elapsed = time.perf_counter() - start
        print(f"Generated optimal trajectory in {elapsed} seconds.")

        # Extract the features from that optimal trajectory:
        features = np.empty((self.w_dim,))
        for i in range(optimal_trajectory[0].shape[0]):
            features += self.features(
                optimal_trajectory[1][i, :],
                optimal_trajectory[0][i, :],  # added ', :'
            )

        features = features / optimal_trajectory[0].shape[0]
        optimal_trajectory_final = Trajectory(optimal_trajectory, features)
        self.state = self.reset(start_state)
        return optimal_trajectory_final

    def all_actions(self):
        """Return continuous action-space."""
        return self._controls_matrix

    def available_actions(self, current_state):
        """Return the actions available in current_state."""
        if self.is_terminal_state(current_state):
            return None
        else:
            return self.all_actions()

    def next_state(self, current_state, action):
        """Return effect on state if action taken in current_state."""
        if type(current_state) is int:
            self._seed = current_state
            self._state = self._rng.random((self._state_vector_size, 1))
            current_state = self._state
        action = np.array(action)
        action = action * self._controls_per_state
        if action.shape != current_state.shape:
            action = action.reshape(current_state.shape)
            s_prime = current_state + action
        else:
            s_prime = current_state + action
        return s_prime

    def is_terminal_state(self, current_state):
        """Check if current_state signals execution completion."""
        if np.all(np.isclose(current_state, self._goal_state)):
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
