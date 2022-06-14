"""A simple Linear Dynamical System domain."""
import time
from pathlib import Path
from typing import Union

import numpy as np
from inquire.environments.environment import Environment
from inquire.utils.datatypes import Range, Trajectory
from inquire.utils.sampling import TrajectorySampling


class PatsLinearDynamicalSystem(Environment):
    """A domain-agnostic linear dynamical system."""

    def __init__(
        self,
        seed: int = None,
        timesteps: int = 90,
        state_vector_size: int = 4,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: np.ndarray = 2000,
        output_path: str = str(Path.cwd()) + "/output/LinearDynamicalSystem/",
        verbose: bool = True,
    ):
        """Initialize the Linear Dynamical System.

        We instantiate the generic LDS model:

            dx/dt = Ax(t) + Bu(t),

        with dynamics matrix A, state vector x, controls matrix B, and controls
        vector u. Assume A = B = I_{state_vector_size} to simplify the system:

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
        """
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._state_multiplier = 50
        self._w_dim = 2 * state_vector_size
        self.trajectory_length = trajectory_length
        self._output_path = output_path
        self._state_vector_size = state_vector_size
        self._optimal_trajectory_iterations = optimal_trajectory_iterations
        self._timesteps = timesteps
        self._verbose = verbose

        # Randomly select goal state:
        self._goal_state = self._state_multiplier * self._rng.random(
            size=(self._state_vector_size, 1)
        )
        self._controls_bounds = [(-1, 1)] * state_vector_size
        self._lower_bound = np.full(
            shape=(state_vector_size,), fill_value=self._controls_bounds[0][0]
        )
        self._upper_bound = np.full(
            shape=(state_vector_size,), fill_value=self._controls_bounds[0][1]
        )
        # Apply a control to each state-element for timesteps time:
        self._controls_vector = np.zeros((self._state_vector_size,))
        self._action_range = Range(
            self._lower_bound,
            np.ones_like(self._lower_bound),
            self._upper_bound,
            np.ones_like(self._upper_bound),
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

    @property
    def controls_bounds(self) -> list:
        """Return the bounds of each control parameter."""
        return self._controls_bounds

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate random state vector."""
        generated = self._rng.random(size=(self._state_vector_size,))
        return generated

    def generate_random_reward(self, random_state) -> np.ndarray:
        """Generate random weights with L2 norm = 1."""
        generated = self._rng.uniform(low=-1, high=1, size=(self.w_dim(),))
        generated = generated / np.linalg.norm(generated)
        return generated

    def features_from_trajectory(
        self, trajectory: list, use_mean: bool = True
    ) -> np.ndarray:
        """Compute the features across an entire trajectory."""
        feats = np.zeros((self.w_dim(),))
        for i in range(trajectory.states.shape[0]):
            feats += self.features(
                trajectory.actions[i, :], trajectory.states[i, :]
            )
        if use_mean:
            return feats / trajectory.states.shape[0]
        else:
            # Always use the means of the actions:
            feats[4:] = feats[4:] / trajectory.states.shape[0]
            return feats

    def features(
        self,
        action: np.ndarray,
        state: Union[int, np.ndarray],
        normalize: bool = True,
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
        if normalize:
            # Normalize the element-wise difference between current state and
            # goal state to be in (-1, 1). Note: Actions already in (-1,1):
            s_diff = np.abs(state - self._goal_state) / (
                2 * self._state_multiplier
            )
        else:
            s_diff = np.abs(state - self._goal_state)
        latest_features = np.concatenate((s_diff, action)).reshape(-1, 1)
        return latest_features.squeeze()

    def trajectory_rollout(
        self, start_state: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Generate trajectory from controls."""
        controls = actions.reshape(-1, self._controls_vector.shape[0])
        # We need to perform each control for some timespan
        # t = self._timesteps_per_state:
        time_adjusted_controls = controls.repeat(
            self._timesteps_per_state, axis=0
        )
        states = np.empty_like(time_adjusted_controls)
        current_state = start_state
        for i, u in enumerate(time_adjusted_controls):
            states[i, :] = current_state
            if self.is_terminal_state(states[i, :]):
                # Reached goal in fewer controls than provided; adjust
                # accordingly:
                states = states[: i + 1, :]
                time_adjusted_controls = time_adjusted_controls[: i + 1, :]
                trajectory = Trajectory(
                    states=states, actions=time_adjusted_controls, phi=None
                )
                trajectory.phi = self.features_from_trajectory(trajectory)
                return trajectory
            else:
                current_state = current_state + u
        trajectory = Trajectory(
            states=states, actions=time_adjusted_controls, phi=None
        )
        trajectory.phi = self.features_from_trajectory(trajectory)
        return trajectory

    def optimal_trajectory_from_w(
        self, start_state: np.ndarray, w: np.ndarray
    ):
        """Compute the optimal trajectory to goal given weights w.

        ::inputs:
            ::start_state: A state with which we reset the environment.
            ::w: An array of weights.
        """
        start = time.perf_counter()
        # Find some controls given start_state and weights w:
        rand = np.random.RandomState(0)
        samples = TrajectorySampling.uniform_sampling(
            start_state,
            None,
            self,
            rand,
            self.trajectory_length,
            self._optimal_trajectory_iterations,
            {},
        )
        rewards = [(w @ s.phi.T).squeeze() for s in samples]
        optimal_trajectory = samples[np.argmax(rewards)]
        elapsed = time.perf_counter() - start
        if self._verbose:
            print(
                "Finished generating optimal trajectory in "
                f"{elapsed:.3f} seconds."
            )
        return optimal_trajectory

    def action_space(self) -> Range:
        """Return the environment's range of possible actions."""
        return self._action_range

    def is_terminal_state(self, current_state):
        """Check if current_state signals execution completion."""
        if np.all(np.isclose(current_state, self._goal_state)):
            return True
        else:
            return False

    def w_dim(self):
        """Return dimensionality of feature-space."""
        return self._w_dim

    def state_space(self):
        """Return dimensionality of observation space."""
        return self.state_space_dim

    def distance_between_trajectories(self, a, b):
        """Stub."""
        return None

    def visualize_trajectory(self) -> None:
        """Stub."""
        pass
