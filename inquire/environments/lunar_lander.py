"""A Lunar Lander environment compatible with Inquire framework."""
import sys
import time
from pathlib import Path
from typing import Union

import dtw
import gym
import numpy as np
from inquire.environments.environment import Environment
from inquire.utils.datatypes import CachedSamples, Range, Trajectory
from inquire.utils.sampling import TrajectorySampling


class LunarLander(Environment):
    """An instance of OpenAI's LunarLanderContinuous domain."""

    def __init__(
        self,
        name: str = "LunarLanderContinuous-v2",
        seed: int = None,
        timesteps: int = 450,
        frame_delay_ms: int = 20,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: int = 1000,
        output_path: str = str(Path.cwd()) + "/output/LunarLander/",
        verbose: bool = False,
        include_feature_biases: bool = False,
    ):
        """
        Initialize OpenAI's LunarLander domain.

        ::inputs:
            ::name: Name of the environment to make.
            ::seed: Seed used to generate the world for demonstrations and
                    preferences.
            ::timesteps: Length of trajectory to be watched (centiseconds).
            ::frame_delay_ms: Delay for smoother animation.
            ::trajectory_length: The number of discrete states and controls
                                 in a trajectory.
            ::optimal_trajectory_iters: The number of control samples to test.
        """
        # Instantiate the openai gym environment:
        self.env = gym.make(name)
        self.seed = seed
        self.reset()
        self.optimal_trajectory_iters = optimal_trajectory_iterations
        self.output_path = output_path
        self.verbose = verbose
        self._include_feature_biases = include_feature_biases

        self.control_size = self.env.action_space.shape[0]
        self.timesteps = timesteps
        self.frame_delay_ms = frame_delay_ms
        self.trajectory_length = trajectory_length
        try:
            assert timesteps % self.trajectory_length == 0
            self.timesteps_per_state = int(timesteps / self.trajectory_length)
        except AssertionError:
            print(
                f"Timesteps ({timesteps}) isn't divisible by "
                f"trajectory length ({self.trajectory_length})."
            )
            exit()
        self.reward_rang = Range(
            np.array([-1]), np.array([1]), np.array([1]), np.array([1])
        )
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        self.action_rang = Range(
            action_low,
            np.ones_like(action_low),
            action_high,
            np.ones_like(action_high),
        )
        state_low = self.env.observation_space.low
        state_high = self.env.observation_space.high
        self.state_rang = Range(
            state_low,
            np.ones_like(state_low),
            state_high,
            np.ones_like(state_high),
        )

    def w_dim(self) -> int:
        """Return the dimensionality of the features."""
        return 4

    def action_space(self) -> Range:
        """Return the range of possible actions."""
        return self.action_rang

    def state_space(self) -> Range:
        """Return the range of possible states."""
        return self.state_rang

    def reward_range(self) -> Range:
        """Return the range of possible rewards."""
        return self.reward_rang

    def generate_random_state(self, random_state):
        """Generate random seed with which we reset env."""
        # When fed the same random_state, the environment returns to
        # the same reset state.
        rando = random_state.randint(low=0, high=sys.maxsize)
        return rando

    def generate_random_reward(self, random_state):
        """Randomly generate a weight vector for trajectory features."""
        # reward = np.array([-0.4, 0.4, -0.2, -0.7])
        reward = np.array([0.55, 0.55, 0.41, 0.48])
        # reward = np.random.uniform(-1, 1, 4)
        return reward / np.linalg.norm(reward)

    def reset(self) -> np.ndarray:
        """Reset to a starting-state defined by seed.

        Converted from DemPref codebase. Note: This method only re-seeds the
        LunarLander environment's state; it does NOT re-seed a random number
        generator.
        """
        state = self.env.reset(seed=self.seed)
        return state

    def trajectory_rollout(
        self, start_state: Union[int, CachedSamples], actions: np.ndarray
    ) -> Trajectory:
        """Collect a trajectory from given controls.

        Adapted from DemPref codebase.
        """
        if isinstance(start_state, CachedSamples):
            self.seed = start_state.state
        elif isinstance(start_state, int):
            self.seed = start_state
        obser = self.reset()

        controls = actions
        c = np.array([[0.0] * self.control_size] * self.timesteps)
        j = 0
        for i in range(self.trajectory_length):
            c[
                i * self.trajectory_length : (i + 1) * self.trajectory_length
            ] = [controls[j + i] for i in range(self.control_size)]
            j += self.control_size

        # obser = self.reset()
        s = [obser]
        for i in range(self.timesteps):
            try:
                results = self.env.step(c[i])
            except:
                print("Caught unstable simulation; skipping.")
                print("Controls which caused this error:\n " f"{c[i]}")
                return None
            obser = results[0]
            s.append(obser)  # A list of np arrays
            if results[2]:
                break
        if len(s) <= self.timesteps:
            c = c[: len(s), :]
        else:
            c = np.append(c, [np.zeros(self.control_size)], axis=0)
        t = Trajectory(states=np.array(s), actions=np.array(c), phi=None)
        t.phi = self.features_from_trajectory(t, use_mean=True)
        return t

    def optimal_trajectory_from_w(
        self, start_state: Union[int, CachedSamples], weights: np.ndarray
    ) -> Trajectory:
        """Optimize a trajectory defined by start_state and weights."""
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

    def features_from_trajectory(
        self, trajectory: Trajectory, use_mean: bool = False
    ) -> np.ndarray:
        """Compute the features across an entire trajectory."""
        feats = np.zeros((trajectory.states.shape[0], self.w_dim()))
        for i in range(trajectory.states.shape[0]):
            last_state = i + 1 == trajectory.states.shape[0]
            feats[i, :] = self.feature_fn(
                trajectory.states[i], at_last_state=last_state
            )
        # Set the initial velocity equal to 0 (which is exp(0)=1):
        feats[0, -2] = 1
        if use_mean:
            # Find the average value of the first three features:
            feats_final = np.mean(feats[:, :-1], axis=0)
            # Tack on the "distance from goal" feature unique to the last
            # state:
            feats_final = np.append(feats_final, feats[-1, -1])
        else:
            feats_final = feats.sum(axis=0)
        return feats_final

    def feature_fn(self, state: np.ndarray, at_last_state: bool = False) -> np.ndarray:
        """Get a trajectory's features.

        The state attributes:
              s[0] is the horizontal coordinate
              s[1] is the vertical coordinate
              s[2] is the horizontal speed
              s[3] is the vertical speed
              s[4] is the angle
              s[5] is the angular speed
              s[6] 1 if first leg has contact, else 0
              s[7] 1 if second leg has contact, else 0

        The landing pad is always at coordinate (0,0).
        """

        def dist_from_landing_pad(state: np.ndarray):
            """Compute distance from the landing pad (which is at (0,0)).

            Left  = positive
            Right = negative
            """
            if self._include_feature_biases:
                return 15 * np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))
            else:
                return np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        def lander_angle(state: np.ndarray):
            """Compute lander's angle w.r.t. ground.

            Angle = 0 when lander is upright.
            """

            if self._include_feature_biases:
                return 15 * np.exp(-np.abs(state[4]))
            else:
                return np.exp(-np.abs(state[4]))

        def velocity(state: np.ndarray):
            """Compute the lander's velocity."""
            if self._include_feature_biases:
                return 10 * np.exp(-np.sqrt(state[2] ** 2 + state[3] ** 2))
            else:
                return np.exp(-np.sqrt(state[2] ** 2 + state[3] ** 2))

        def final_position(state: np.ndarray):
            """Lander's final state position."""
            if self._include_feature_biases:
                return 30 * np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))
            else:
                return np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        # Compute the features of this new state:
        phi = np.stack(
            [
                dist_from_landing_pad(state),
                lander_angle(state),
                velocity(state),
                0,
            ]
        )
        # If we've reached the terminal state, compute the corresponding
        # feature:
        if at_last_state:
            phi[-1] = final_position(state)
        return phi

    def distance_between_trajectories(self, a, b):
        a_points = [[state[0], state[1]] for state in a.states]
        b_points = [[state[0], state[1]] for state in b.states]
        alignment = dtw.dtw(a_points, b_points)
        return alignment.normalizedDistance

    def visualize_trajectory(
        self, start_state, trajectory, frame_delay_ms: int = 20
    ):
        """Visualize a trajectory defined by start_state."""
        if isinstance(start_state, CachedSamples):
            self.seed = start_state.state
        elif isinstance(start_state, int):
            self.seed = start_state
        self.reset()
        for i in range(len(trajectory.actions)):
            self.env.render()
            a = trajectory.actions[i]
            observation, reward, done, info = self.env.step(a)
            time.sleep(frame_delay_ms / 1000)
            if done:
                break
        self.env.close()
