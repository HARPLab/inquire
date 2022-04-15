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

        def reward_fn(controls: np.ndarray, weights: np.ndarray) -> float:
            """Return reward for given controls and weights."""
            trajectory = self.run(controls)
            features = np.empty((self._number_of_features, 1))
            for i, s in enumerate(trajectory):
                features[i, 0] += self.features(
                    trajectory[0][i, :], trajectory[1][i, :]
                )

            features = features.mean()
            rwd = -weights.T @ features
            return rwd

        self._seed = start_state
        self._state = self.known_reset(start_state)

        low = [x[0] for x in self.control_bounds] * self._trajectory_length
        high = [x[1] for x in self.control_bounds] * self._trajectory_length
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
                    low=low,
                    high=high,
                    size=self.control_size * trajectory_length,
                ),
                args=(self, weights),
                bounds=self.control_bounds * trajectory_length,
                approx_grad=True,
                # maxfun=1000,
                # maxiter=100,
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
                "state seed": start_state,
                "state_0": optimal_trajectory[0][:, 0],
                "state_1": optimal_trajectory[0][:, 1],
                "state_2": optimal_trajectory[0][:, 2],
                "state_3": optimal_trajectory[0][:, 3],
                "state_4": optimal_trajectory[0][:, 4],
                "state_5": optimal_trajectory[0][:, 5],
                "state_6": optimal_trajectory[0][:, 6],
                "state_7": optimal_trajectory[0][:, 7],
            }
        )
        # current_time = time.localtime()
        # TODO Consider pickling:
        # if self.verbose:
        #    print("Saving trajectory ...")
        # df.to_csv(
        #    self.output_path
        #    + time.strftime("%m:%d:%H:%M:%S_", current_time)
        #    + f"_weights_{weights[0]:.2f}_{weights[1]:.2f}_{weights[2]:.2f}_{weights[3]:.2f}_"
        #    + ".csv"
        # )

        # Extract the features from that optimal trajectory:
        features = np.zeros((optimal_trajectory[0].shape[0], self.w_dim))
        for i in range(optimal_trajectory[0].shape[0]):
            if i + 1 == optimal_trajectory[0].shape[0]:
                f = self.feature_fn(
                    optimal_trajectory[1][i],
                    optimal_trajectory[0][i],
                    at_last_state=True,
                )
            else:
                f = self.feature_fn(
                    optimal_trajectory[1][i], optimal_trajectory[0][i]
                )
            features[i, :] = f

        # Set the initial velocity equal to 0:
        features[0, -2] = 0
        phi_total = np.mean(features[:, :-1], axis=0)
        phi_total = np.append(phi_total, features[-1, -1])
        optimal_trajectory_final = Trajectory(optimal_trajectory, phi_total)
        self.state = self.known_reset(start_state)
        return optimal_trajectory_final

    def all_actions(self):
        """Return continuous action-space."""
        return self._controls_vector

    def available_actions(self, current_state):
        """Return the actions available in current_state."""
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
