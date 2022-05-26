"""A Lunar Lander environment compatible with Inquire framework."""
import time
from pathlib import Path
from typing import List, Union

import gym

from inquire.environments.gym_wrapper_environment import GymWrapperEnvironment
from inquire.interactions.feedback import Trajectory

import numpy as np

import pandas as pd

import scipy.optimize as opt


class LunarLander(GymWrapperEnvironment):
    """An instance of OpenAI's LunarLanderContinuous domain."""

    def __init__(
        self,
        name: str = "LunarLanderContinuous-v2",
        seed: int = None,
        num_features: int = 4,
        timesteps: int = 450,
        frame_delay_ms: int = 20,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: int = 50,
        output_path: str = str(Path.cwd()) + "/output/lunar_lander/",
        verbose: bool = False,
        save_weights: bool = False,
        save_trajectory: bool = False,
    ):
        """
        Initialize OpenAI's LunarLander domain.

        ::inputs:
            ::name: Name of the environment to make.
            ::seed: Seed used to generate the world for demonstrations and
                    preferences.
            ::num_features: Number of features in a trajectory.
            ::timesteps: Length of trajectory to be watched (centiseconds).
            ::frame_delay_ms: Delay for smoother animation.
            ::trajectory_length: The number of discrete states and controls
                                 in a trajectory.

        Bulk of code converted from DemPref/domain.py: run(), simulate(),
        reset(), features().
        """
        # Instantiate the openai gym environment:
        self.env = gym.make(name)
        # Wrap the environment in the functionality pertinent to Inquire and
        # initiate starting state:
        super(LunarLander, self).__init__(
            name, self.env, self.optimal_trajectory_fn, output_path
        )
        self.seed = seed
        self.env.reset(seed=self.seed)
        self.w_dim = num_features
        self.optimal_trajectory_iters = optimal_trajectory_iterations
        self.output_path = output_path
        self.verbose = verbose
        self.save_weights = save_weights
        self.save_trajectory = save_trajectory
        # Carry on with (most of) the authors' original instantiation:
        self.control_size = self.env.action_space.shape[0]
        self.control_bounds = [
            (self.env.action_space.low[i], self.env.action_space.high[i])
            for i in range(self.env.action_space.shape[0])
        ]
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

    def reset(self, seed: int = None) -> np.ndarray:
        """Reset to a starting-state defined by seed.

        Converted from DemPref codebase. Note: This method only re-seeds the
        LunarLander environment's state; it does NOT re-seed a random number
        generator.
        """
        if not seed:
            seed = self.seed
        # self.env.seed(seed)
        state = self.env.reset(seed=seed)
        return state

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
            raw_trajectory, use_mean=True
        )

        full_trajectory = Trajectory(raw_trajectory, trajectory_phis)
        return full_trajectory

    def run(self, controls: np.ndarray) -> List[np.array]:
        """Collect a trajectory from given controls.

        Converted from DemPref codebase.
        """
        c = np.array([[0.0] * self.control_size] * self.timesteps)
        j = 0
        for i in range(self.trajectory_length):
            c[
                i * self.trajectory_length : (i + 1) * self.trajectory_length
            ] = [controls[j + i] for i in range(self.control_size)]
            j += self.control_size

        # Note the reset-seed is assigned in optimal_trajectory()
        obser = self.reset()
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
        return [np.array(s), np.array(c)]

    def optimal_trajectory_fn(
        self,
        start_state: int,
        weights: np.ndarray,
        trajectory_length: int = 10,
    ) -> np.ndarray:
        """Optimize a trajectory defined by start_state and weights."""

        def reward_fn(
            controls: np.ndarray,
            domain: GymWrapperEnvironment,
            weights: np.ndarray,
        ):
            """One-step reward function.

            ::inputs:
              ::controls: thruster velocities
              ::weights: weight for reward function
            """
            t = self.run(controls)
            feats = self.features_from_trajectory(t, use_mean=True)
            reward = (weights @ feats.T).squeeze()
            # Negate reward to minimize via BFGS:
            return -reward

        # Always set the seed and reset environment:
        self.seed = start_state
        self.reset(start_state)

        low = [x[0] for x in self.control_bounds] * trajectory_length
        high = [x[1] for x in self.control_bounds] * trajectory_length
        optimal_ctrl = None
        opt_val = np.inf
        start = time.perf_counter()
        # Find the optimal controls given the start state and weights:
        for _ in range(self.optimal_trajectory_iters):
            if self.verbose:
                print(
                    f"Beginning optimization iteration {_ + 1} of "
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

        # TODO Consider pickling instead of:
        current_time = time.localtime()
        if self.save_weights:
            df = pd.DataFrame(
                {
                    "state seed": start_state,
                    "weight_0": weights[0],
                    "weight_1": weights[1],
                    "weight_2": weights[2],
                    "weight_3": weights[3],
                }
            )
            df.to_csv(
                self.output_path
                + time.strftime("%m:%d:%H:%M:%S_", current_time)
                + f"{self.__repr__()}_weights"
            )

        if self.save_trajectory:
            print("Saving trajectory ...")
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
            df.to_csv(
                self.output_path
                + time.strftime("%m:%d:%H:%M:%S_", current_time)
                + f"_weights_{weights[0]:.2f}_{weights[1]:.2f}"
                + f"_{weights[2]:.2f}_{weights[3]:.2f}"
                + ".csv"
            )
        optimal_phi = self.features_from_trajectory(
            optimal_trajectory, use_mean=True
        )
        optimal_trajectory_final = Trajectory(optimal_trajectory, optimal_phi)
        self.reset(start_state)
        return optimal_trajectory_final

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
        feats = np.zeros((trajectory[0].shape[0], self.w_dim))
        for i in range(trajectory[0].shape[0]):
            if i + 1 == trajectory[0].shape[0]:
                f = self.feature_fn(
                    action=trajectory[1][i],
                    state=trajectory[0][i],
                    at_last_state=True,
                )
            else:
                f = self.feature_fn(
                    action=trajectory[1][i], state=trajectory[0][i]
                )
            feats[i, :] = f
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

    def feature_fn(
        self, action, state, at_last_state: bool = False
    ) -> np.ndarray:
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
            return np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        def lander_angle(state: np.ndarray):
            """Compute lander's angle w.r.t. ground.

            Angle = 0 when lander is upright.
            """
            return np.exp(-np.abs(state[4]))

        def velocity(state: np.ndarray):
            """Compute the lander's velocity."""
            return np.exp(-np.sqrt(state[2] ** 2 + state[3] ** 2))

        def final_position(state: np.ndarray):
            """Lander's final state position."""
            return np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        # Compute the features of this new state:
        phi = np.stack(
            [
                self.timesteps_per_state * dist_from_landing_pad(state),
                self.timesteps_per_state * lander_angle(state),
                self.timesteps_per_state * velocity(state),
                0,
            ]
        )
        # If we've reached the terminal state, compute the corresponding
        # feature:
        if at_last_state:
            phi[-1] = final_position(state)
        return phi
