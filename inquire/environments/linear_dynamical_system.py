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
        timesteps: int = 90,
        state_vector_size: int = 4,
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

        In this system, features are the elements of the state and the controls
        applied to each element in the state.

        ::inputs:
            ::seed: Used to seed a random number generator. Default 'None'
                    assumes pseudo-random behavior.
            ::timesteps: The number of seconds it takes to execute a single
                         control.
            ::state_vector_size: Number of (arbitrary) elements that define a
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
        self._rng = np.random.default_rng(self._seed)
        self.w_dim = 2 * state_vector_size
        self._trajectory_length = trajectory_length
        self._output_path = output_path
        self._state_vector_size = state_vector_size
        self._optimal_trajectory_iterations = optimal_trajectory_iterations
        self._timesteps = timesteps
        self._verbose = verbose

        # Randomly select goal state:
        self._goal_state = 50 * self._rng.random(
            size=(self._state_vector_size, 1)
        )
        # self._controls_bounds = np.array([[-1, 1]]).repeat(
        #    state_vector_size, axis=0
        # )
        self._controls_bounds = [(-1, 1)] * state_vector_size
        self._optimizers_controls_bounds = (
            self._controls_bounds * self._trajectory_length
        )
        # self._optimizers_controls_bounds = self._controls_bounds.repeat(
        #    self._trajectory_length, axis=0
        # )
        self._lower_bound = [
            x[0] for x in self._controls_bounds
        ] * self._trajectory_length
        self._upper_bound = [
            x[1] for x in self._controls_bounds
        ] * self._trajectory_length
        # Apply a control to each state-element for timesteps time:
        self._controls_vector = np.zeros((self._state_vector_size,))
        self._controls_matrix = np.full(
            (self._state_vector_size, control_space_discretization),
            fill_value=np.linspace(
                -1, 1, control_space_discretization, endpoint=True
            ),
        )

        try:
            assert timesteps % trajectory_length == 0
            self._timesteps_per_state = int(timesteps / trajectory_length)
        except AssertionError:
            print(
                f"Timesteps ({timesteps}) isn't divisible by "
                f"trajectory length ({trajectory_length})."
            )
            exit()

    def __repr__(self) -> str:
        """Return the class' representative string."""
        return f"{self.__class__.__name__}"

    def reset(self, start_state: np.ndarray = None) -> None:
        """Reset the environment to start_state."""
        if start_state is not None:
            self._start_state = start_state
        self._state = self._start_state

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate random state vector."""
        generated = self._rng.random(size=(self._state_vector_size, 1))
        return generated

    def generate_random_reward(self, random_state) -> np.ndarray:
        """Generate random weights with L2 norm = 1."""
        generated = self._rng.uniform(low=-1, high=1, size=(self.w_dim,))
        generated = generated / np.linalg.norm(generated)
        return generated

    def features_from_trajectory(
        self,
        trajectory_input: list,
        controls_as_input: bool = False,
        use_mean: bool = False,
    ) -> np.ndarray:
        """Compute the features across an entire trajectory."""
        if controls_as_input:
            trajectory = self.run(trajectory_input)
        else:
            trajectory = trajectory_input
        feats = np.zeros((self.w_dim,))
        for i in range(trajectory[0].shape[0]):
            feats += self.features(trajectory[1][i, :], trajectory[0][i, :])
        if use_mean:
            return feats / trajectory[0].shape[0]
        else:
            # Always use the means of the actions:
            feats[4:] = feats[4:] / trajectory[0].shape[0]
            return feats

    def features(
        self, action: np.ndarray, state: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Compute features of state.

        ::inputs:
            ::state: The CURRENT state.
            ::action: The control that caused the transition to state.
        """
        if action is None:
            action = np.zeros_like(state)
        action = np.array(action).reshape(-1, 1)
        state = state.reshape(-1, 1)
        # Compute a simple distance metric to keep feature-count
        # proportional to number of elements which define the state:
        s_diff = np.abs(state - self._goal_state)
        latest_features = np.exp(
            -np.concatenate((s_diff, np.abs(action).reshape(-1, 1)))
        )
        return latest_features.squeeze()

    def trajectory_from_states(self, sample: Union[list, np.ndarray], features) -> Trajectory:
        """Convert list of state-action pairs to a Trajectory."""
        if type(sample) == list:
            sample = np.array(sample, dtype=object)
        if sample[0, 0] is None:
            sample = sample[1:, :]

        controls = np.hstack(sample[:, 0])
        # Get all state-action pairs:
        raw_trajectory = self.run(controls)

        # Get the features from those state-action pairs:
        trajectory_phis = self.features_from_trajectory(
            raw_trajectory, use_mean=False
        )

        full_trajectory = Trajectory(raw_trajectory, trajectory_phis)
        return full_trajectory

    def run(self, controls: np.ndarray) -> np.ndarray:
        """Generate trajectory from controls."""
        controls = controls.reshape(-1, self._controls_vector.shape[0])
        trajectory = np.empty(
            (
                int(self._trajectory_length * self._timesteps_per_state),
                self._state_vector_size,
            )
        )
        # We need to perform each control for some timespan
        # t = self._timesteps_per_state:
        time_adjusted_controls = controls.repeat(
            self._timesteps_per_state, axis=0
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
            else:
                self._state = self._state + u.reshape(-1, 1)
        return [trajectory, time_adjusted_controls]

    def optimal_trajectory_from_w(
        self, start_state: np.ndarray, w: np.ndarray
    ):
        """Compute the optimal trajectory to goal given weights w.

        ::inputs:
            ::start_state: A state with which we reset the environment.
            ::w: An array of weights.
        """
        # Always set the seed and reset environment:
        self._start_state = start_state
        self.reset()

        def reward_fn(
            controls: np.ndarray, domain: Environment, weights: np.ndarray
        ) -> float:
            """Return reward for given controls and weights."""
            trajectory = self.run(controls)
            features = self.features_from_trajectory(
                trajectory, use_mean=False
            )
            reward = (features.T @ w).squeeze()
            # Negate reward to minimize via BFGS:
            return -reward

        optimal_ctrl = None
        opt_val = np.inf
        start = time.perf_counter()
        # Find the optimal controls given start_state and weights w:
        for _ in range(self._optimal_trajectory_iterations):
            inner_start = time.perf_counter()
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
                maxfun=1000,
                maxiter=100,
            )
            if temp_result[1] < opt_val:
                optimal_ctrl = temp_result[0]
                opt_val = temp_result[1]
            inner_elapsed = time.perf_counter() - inner_start
            if self._verbose:
                print(
                    f"Iteration {_+1} completed in {inner_elapsed:.3f} "
                    "seconds."
                )
        elapsed = time.perf_counter() - start
        if self._verbose:
            print(
                "Finished generating optimal trajectory in "
                f"{elapsed:.3f} seconds."
            )

        optimal_trajectory = self.run(optimal_ctrl)
        elapsed = time.perf_counter() - start

        # Extract the features from that optimal trajectory:
        optimal_features = self.features_from_trajectory(
            optimal_trajectory, use_mean=False
        )
        if self._verbose:
            print(f"Latest features:\n{optimal_features}.")
        optimal_trajectory_final = Trajectory(
            optimal_trajectory, optimal_features
        )
        self.reset()
        return optimal_trajectory_final

    def all_actions(self):
        """Return continuous action-space."""
        return self._controls_matrix

    def available_actions(self, current_state):
        """Return the actions available in current_state."""
        if self.is_terminal_state(current_state):
            return [None] * self._state_vector_size
        else:
            return self.all_actions()

    def next_state(self, current_state, action):
        """Return effect on state if action taken in current_state."""
        action = np.array(action).reshape(current_state.shape)
        action = action.repeat(self._timesteps_per_state, axis=1)
        s_prime = np.array(current_state, copy=True)
        for i in range(action.shape[1]):
            s_prime = s_prime + action[:, i].reshape(s_prime.shape)
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
