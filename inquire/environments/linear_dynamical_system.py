"""A submodule to define the Linear Dynamical System environment."""
from typing import Union
import dtw

import numpy as np
from inquire.environments.environment import Environment
from inquire.utils.datatypes import Trajectory, Range, CachedSamples
from inquire.utils.sampling import TrajectorySampling


class LinearDynamicalSystem(Environment):
    """A domain-agnostic linear dynamical system."""

    def __init__(
        self,
        timesteps: int = 90,
        state_vector_size: int = 4,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: np.ndarray = 100,
        control_space_discretization: int = 2000,
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
        self.weight_dim = 2 * state_vector_size
        self.trajectory_length = trajectory_length
        self._state_vector_size = state_vector_size
        self._optimal_trajectory_iterations = optimal_trajectory_iterations
        self._timesteps = timesteps
        self._verbose = verbose
        self.optimal_trajectory_iters = optimal_trajectory_iterations

        self.action_rang = Range(
            -1 * np.ones(state_vector_size),
            np.ones(state_vector_size),
            np.ones(state_vector_size),
            np.ones(state_vector_size)
        )
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

    def w_dim(self):
        return self.weight_dim

    def action_space(self) -> Range:
        """Return the range of possible actions."""
        return self.action_rang

    def state_space(self) -> Range:
        """Return the range of possible states."""
        return None

    def reward_range(self) -> Range:
        """Return the range of possible rewards."""
        return None

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate random init and goal state vectors."""
        init_state = 15 * random_state.random(size=(self._state_vector_size,1))
        return init_state

    def generate_random_reward(self, random_state) -> np.ndarray:
        """Generate random weights with L2 norm = 1."""
        state_reward = np.ones(int(self.weight_dim/2),)
        action_reward = random_state.normal(0, 0.25, size=(self.weight_dim-state_reward.shape[0],))
        generated = np.concatenate([state_reward, action_reward])
        generated = generated / np.linalg.norm(generated)
        return generated

    def features_from_trajectory(self, trajectory: Trajectory, use_mean: bool = False) -> np.ndarray:
        feats = np.zeros((self.weight_dim,))
        for i in range(trajectory.states.shape[0]):
            feats += self.features(trajectory.actions[i, :], trajectory.states[i, :])
        if use_mean:
            return feats / trajectory.states.shape[0]
        else:
            # Always use the means of the actions:
            feats[4:] = feats[4:] / trajectory.states.shape[0]
            return feats

    def features(
            self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
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
        s_diff = np.abs(state)
        latest_features = np.exp(
            -np.concatenate((s_diff, np.abs(action).reshape(-1, 1)))
        )
        return latest_features.squeeze()

    def trajectory_rollout(self, start_state: Union[int, CachedSamples], actions: np.ndarray) -> Trajectory:
        if isinstance(start_state, CachedSamples):
            state = start_state.state
        else:
            state = start_state
        """Generate trajectory from controls."""
        controls = actions.reshape(-1, self._controls_vector.shape[0])
        states = np.empty(
            (
                int(self.trajectory_length * self._timesteps_per_state),
                self._state_vector_size,
            )
        )
        # We need to perform each control for some timespan
        # t = self._timesteps_per_state:
        time_adjusted_controls = controls.repeat(
            self._timesteps_per_state, axis=0
        )
        for i, u in enumerate(time_adjusted_controls):
            states[i, :] = state.squeeze()
            if np.all(np.isclose(states[i, :], np.zeros_like(states[i,:]))):
                # Reached goal in fewer controls than provided; adjust
                # accordingly:
                states = states[: i + 1, :]
                time_adjusted_controls = time_adjusted_controls[: i + 1, :]
                traj = Trajectory(states=states, actions=time_adjusted_controls, phi=None)
                traj.phi = self.features_from_trajectory(traj, use_mean=True)
                return traj
            else:
                state = state + u.reshape(-1, 1)
        traj = Trajectory(states=states, actions=time_adjusted_controls, phi=None)
        traj.phi = self.features_from_trajectory(traj, use_mean=True)
        return traj

    def optimal_trajectory_from_w(
        self, start_state: Union[int, CachedSamples], weights: np.ndarray
    ) -> Trajectory:

        """Compute the optimal trajectory to goal given weights w.

        ::inputs:
            ::start_state: A state with which we reset the environment.
            ::w: An array of weights.
        """
        rand = np.random.RandomState(0)
        samples = TrajectorySampling.uniform_sampling(
            start_state,
            None,
            self,
            rand,
            self.trajectory_length,
            self.optimal_trajectory_iters,
            {},
        )
        rewards = [(weights @ s.phi.T).squeeze() for s in samples]
        opt_traj = samples[np.argmax(rewards)]
        return opt_traj

    def distance_between_trajectories(self, a, b):
        a_points = [[state[0],state[1]] for state in a.states]
        b_points = [[state[0],state[1]] for state in b.states]
        alignment = dtw.dtw(a_points, b_points)
        return alignment.normalizedDistance

    def visualize_trajectory(
        self, start_state, trajectory, frame_delay_ms: int = 20
    ):
        return None
