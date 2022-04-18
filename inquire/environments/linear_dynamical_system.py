"""A submodule to define the Linear Dynamical System environment."""
import time
from pathlib import Path

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
                                 a more apt term in this context.
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
        self._state = np.zeros((state_vector_size, 1))
        # Randomly select goal state:
        self._goal_state = self._rng.integers(
            low=0, high=100, size=(self._state_vector_size, 1)
        )
        self._controls_bounds = np.array([[-1, 1]]).repeat(
            state_vector_size, axis=0
        )
        self._lower_bound = [
            x[0] for x in self.control_bounds
        ] * self._trajectory_length
        self._upper_bound = [
            x[1] for x in self.control_bounds
        ] * self._trajectory_length
        # Apply a control to each state-element for trajectory_duration time:
        self._controls_vector = np.array(
            [self._trajectory_duration, self._state_vector_size]
        )
        self._controls_matrix = np.full(
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

    def reset(self, seed: int = None) -> None:
        """Reset the environment according to seed.

        Note: Start state is always the 0-vector.
        """
        self._state = np.zeros((self._state_vector_size, 1))
        if seed is None:
            self._rng = np.random.default_rng(self._seed)
        else:
            self._rng = np.random.default_rng(seed)
        self._goal_state = self._rng.integers(
            low=0, high=100, size=(self._state_vector_size, 1)
        )

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate random goal as state vector."""
        rando = self._rng.integers(
            low=0, high=100, size=(self._state_vector_size, 1)
        )
        return rando

    def generate_random_reward(self, random_state) -> np.ndarray:
        """Generate weights of random value between (0, 1)."""
        generated = self._rng.random(size=(self._number_of_features, 1))
        return generated

    def features(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Compute features from taking action when in state.

        ::inputs:
            ::action: The control to apply from state.
        """
        next_state = self.next_state(state, action)

        latest_features = np.empty(
            (self._number_of_features, self._state_vector_size)
        )

        # Extract features--a vector of order-i polynomials:
        for i, f in enumerate(self._number_of_features):
            latest_features[i, :] = next_state ** i
            # latest_features[i, :] = next_state
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
            if self.is_terminal_state(trajectory[i, :]):
                # Reached goal in fewer controls than provided; adjust
                # accordingly:
                trajectory = trajectory[: i + 1, :]
                time_adjusted_controls = time_adjusted_controls[: i + 1, :]
                return [trajectory, time_adjusted_controls]
        return [trajectory, time_adjusted_controls]

    def optimal_trajectory_from_w(self, start_state, w):
        """Compute the optimal trajectory to goal given weights w.

        ::inputs:
            ::start_state: A seed for the environment resets.
            ::w: A set of weights.
        """
        self._seed = start_state
        self.reset(start_state)

        def reward_fn(
            controls: np.ndarray, domain: Environment, weights: np.ndarray
        ) -> float:
            """Return reward for given controls and weights."""
            trajectory = self.run(controls)
            features = np.empty(
                (self._number_of_features, self._state_vector_size)
            )
            for i, s in enumerate(trajectory):
                features[i, :] += self.features(
                    trajectory[1][i, :], trajectory[0][i, :]
                )

            # features = features.sum()
            rwd = 0
            for j, w in enumerate(weights):
                rwd += np.sum(w * features[j, :])
            # rwd = -weights.T @ features
            return -rwd

        optimal_ctrl = None
        opt_val = np.inf
        start = time.perf_counter()
        # Find the optimal controls given the start state and weights:
        for _ in range(self.optimal_trajectory_iters):
            if self.verbose:
                print(
                    f"Beginning optimization iteration {_} of "
                    f"{self.optimal_trajectory_iters}."
                )
            temp_result = opt.fmin_l_bfgs_b(
                reward_fn,
                x0=np.random.uniform(
                    low=self._lower_bound,
                    high=self._upper_bound,
                    size=self._state_vector_size * self._trajectory_length,
                ),
                args=(self, w),
                bounds=self._state_vector_size * self._trajectory_length,
                approx_grad=True,
                maxfun=1000,
                maxiter=100,
            )
            if temp_result[1] < opt_val:
                optimal_ctrl = temp_result[0]
                opt_val = temp_result[1]
        elapsed = time.perf_counter() - start
        if self.verbose:
            print(
                "Finished generating optimal trajectory in "
                f"{elapsed:.4f} seconds."
            )

        optimal_trajectory = self.run(optimal_ctrl)
        df = pd.DataFrame(
            {
                "controller_1": optimal_trajectory[1][:, 0],
                "controller_2": optimal_trajectory[1][:, 1],
                "state_0": optimal_trajectory[0][:, 0],
                "state_1": optimal_trajectory[0][:, 1],
                "state_2": optimal_trajectory[0][:, 2],
                "state_3": optimal_trajectory[0][:, 3],
                "state_4": optimal_trajectory[0][:, 4],
                "state_5": optimal_trajectory[0][:, 5],
                "state_6": optimal_trajectory[0][:, 6],
            }
        )
        current_time = time.localtime()
        if self.verbose:
            print("Saving trajectory ...")
        df.to_csv(
            self.output_path
            + time.strftime("%m:%d:%H:%M:%S_", current_time)
            + f"_weights_{w[0]:.2f}_{w[1]:.2f}_{w[2]:.2f}_{w[3]:.2f}_"
            + ".csv"
        )

        # Extract the features from that optimal trajectory:
        features = np.zeros(
            (optimal_trajectory[0].shape[0], self._number_of_features)
        )
        for i in range(optimal_trajectory[0].shape[0]):
            if i + 1 == optimal_trajectory[0].shape[0]:
                f = self.features(
                    optimal_trajectory[1][i], optimal_trajectory[0][i]
                )
            else:
                f = self.feature(
                    optimal_trajectory[1][i], optimal_trajectory[0][i]
                )
            features[i, :] = f

        phi = np.mean(features, axis=0)
        optimal_trajectory_final = Trajectory(optimal_trajectory, phi)
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
            return self.all_actions

    def next_state(self, current_state, action):
        """Return effect on state if action taken in current_state."""
        if action.shape != self._controls_vector.shape:
            action_all_dimensions = np.full_like(
                self._controls_vector, action.reshape(1, -1)
            )
            next_state = current_state + action_all_dimensions
        else:
            next_state = current_state + action
        return next_state

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
