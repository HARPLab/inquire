"""A submodule to define a Lunar Lander environment compatible with Inquire."""
import time
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import scipy.optimize as opt
from inquire.environments.gym_wrapper_environment import GymWrapperEnvironment
from inquire.interactions.feedback import Trajectory


class LunarLander(GymWrapperEnvironment):
    """An instance of OpenAI's LunarLanderContinuous domain."""

    def __init__(
        self,
        name: str = "LunarLanderContinuous-v2",
        seed: int = None,
        num_features: int = 4,
        time_steps: int = 450,
        frame_delay_ms: int = 20,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: int = 10,
        output_path: str = str(Path.cwd()) + "/output/lunar_lander/",
        verbose: bool = False,
        save_weights: bool = False,
    ):
        """
        Initialize OpenAI's LunarLander domain.

        ::inputs:
            ::name: Name of the environment to make.
            ::seed: Seed used to generate the world for demonstrations and
                    preferences.
            ::num_features: Number of features in a trajectory.
            ::time_steps: Length of trajectory to be watched (centiseconds).
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
            name, self.env, self.optimal_trajectory, output_path
        )
        self.state = self.env.reset()
        self.w_dim = num_features
        self.optimal_trajectory_iters = optimal_trajectory_iterations
        self.output_path = output_path
        self.verbose = verbose
        self.save_weights = save_weights
        # Carry on with (most of) the authors' original instantiation:
        self.seed = seed
        self.env.seed(self.seed)
        self.control_size = self.env.action_space.shape[0]
        self.control_bounds = [
            (self.env.action_space.low[i], self.env.action_space.high[i])
            for i in range(self.env.action_space.shape[0])
        ]
        self.time_steps = time_steps

        self.frame_delay_ms = frame_delay_ms

    def reset(self, seed: int = None):
        """Reset to a known starting-state.

        Converted from DemPref codebase. Note: This method only re-seeds the
        LunarLander environment; it does NOT re-seed a random number generator.
        """
        if not seed:
            seed = self.seed
        self.env.seed(seed)
        state = self.env.reset()
        return state

    def run(self, controls: np.ndarray) -> np.ndarray:
        """Collect a trajectory from given controls.

        Converted from DemPref codebase.
        """
        c = np.array([[0.0] * self.control_size] * self.time_steps)
        num_intervals = len(controls) // self.control_size
        interval_length = self.time_steps // num_intervals

        assert (
            interval_length * num_intervals == self.time_steps
        ), "Number of generated controls must be divisible by total time steps."

        j = 0
        for i in range(num_intervals):
            c[i * interval_length : (i + 1) * interval_length] = [
                controls[j + i] for i in range(self.control_size)
            ]
            j += self.control_size

        # Note the reset-seed is assigned in optimal_trajectory()
        obser = self.reset()
        s = [obser]
        for i in range(self.time_steps):
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
        if len(s) <= self.time_steps:
            c = c[: len(s), :]
        else:
            c = np.append(c, [np.zeros(self.control_size)], axis=0)
        return [np.array(s), np.array(c)]

    def optimal_trajectory(
        self,
        start_state: int,
        weights: np.ndarray,
        trajectory_length: int = 10,
    ) -> np.ndarray:
        """Optimize a trajectory defined by start_state ad weights."""

        def reward_fn(controls, domain, weights):
            """One-step reward function.

            ::inputs:
              ::controls: thruster velocities
              ::weights: weight for reward function
            """
            t = self.run(controls)
            feats = self.features_from_trajectory(t)
            reward = -np.dot(weights, feats.T).squeeze()
            return reward

        self.seed = start_state
        self.state = self.reset(start_state)

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
        current_time = time.localtime()
        # TODO Consider pickling:
        if self.verbose:
            print("Saving trajectory ...")
        if self.save_weights:
            df.to_csv(
                self.output_path
                + time.strftime("%m:%d:%H:%M:%S_", current_time)
                + f"_weights_{weights[0]:.2f}_{weights[1]:.2f}"
                + f"_{weights[2]:.2f}_{weights[3]:.2f}"
                + ".csv"
            )

        # Extract the features from that optimal trajectory:
        optimal_phi = self.features_from_trajectory(optimal_trajectory)
        optimal_trajectory_final = Trajectory(optimal_trajectory, optimal_phi)
        self.state = self.reset(start_state)
        return optimal_trajectory_final

    def features_from_trajectory(
        self, trajectory_input: list, controls_as_input: bool = False
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
                    trajectory[1][i], trajectory[0][i], at_last_state=True
                )
            else:
                f = self.feature_fn(trajectory[1][i], trajectory[0][i])
            feats[i, :] = f
        # Set the initial velocity equal to 0:
        feats[0, -2] = 0
        # Find the average value of the first three features:
        feats_final = np.mean(feats[:, :-1], axis=0)
        # Tack on the "distance from goal" feature unique to the last state:
        feats_final = np.append(feats_final, feats[-1, -1])
        return feats_final

    def feature_fn(
        self, action, state, at_last_state: bool = False
    ) -> np.ndarray:
        """Get the features from a trajectory.

        Note that DemPref's trajectories have multi-dimensional
        arrays--one for states and one for controls.

        [state] is of the form:
          [agent][time][states], where [states] comes from a gym.env.step()

        [control] is of the form:
          [agent][time][controls], where [controls] comes from some call
          to a gym.env.action_space.

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

            Note: Corresponding weight should be negative.
            """
            return -15 * np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        def lander_angle(state: np.ndarray):
            """Compute lander's angle w.r.t. ground.

            Angle = 0 when lander is upright.

            Note: Corresponding weight should be positive.
            """
            return 15 * np.exp(-np.abs(state[4]))

        def velocity(state: np.ndarray):
            """Compute the lander's velocity.

            Note: Corresponding weight should be negative.
            """
            return -10 * np.exp(-np.sqrt(state[2] ** 2 + state[3] ** 2))

        def final_position(state: np.ndarray):
            """Lander's final state position.

            Note: Corresponding weight should be negative.
            """
            return -30 * np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

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
