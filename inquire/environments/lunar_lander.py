import time
import typing

import gym
import numpy as np
import pandas as pd
import scipy.optimize as opt

import gym.utils.play as play

import pickle
from pathlib import Path

from inquire.environments.gym_wrapper_environment import GymWrapperEnvironment
from inquire.interactions.feedback import Trajectory


class LunarLander(GymWrapperEnvironment):
    def __init__(
        self,
        name: str = "LunarLanderContinuous-v2",
        seed: int = 77,
        num_features: int = 4,
        time_steps: int = 250,
        frame_delay_ms: int = 20,
        trajectory_length: int = 10,
        optimal_trajectory_iterations: int = 1,
        output_path: str = Path.cwd() + Path("../../tests/output/"),
    ):
        """
        Initializes OpenAI's LunarLander domain.

        ::inputs:
            ::name: Name of the environment to make.
            ::seed: Seed used to generate the world for demonstrations and
                    preferences.
            ::num_features: Number of features in a trajectory.
            ::time_steps: Time-length of trajectory to be watched.
            ::frame_delay_ms: Delay for animation.
            ::trajectory_length: The number of discrete states and controls
                                 in a trajectory.

        Bulk of code converted from DemPref/domain.py: run(), simulate(),
        reset(), features().
        """
        # 1.) Instantiate the openai gym environment:
        self.env = gym.make(name)
        # 2.) Wrap the environment in the functionality pertinent to
        #     Inquire and initiate starting state:
        super(LunarLander, self).__init__(
            name, self.env, self.optimal_trajectory
        )
        self.state = self.env.reset()
        self.w_dim = num_features
        self.optimal_trajectory_iters = optimal_trajectory_iterations
        # 3.) Carry on with (most of) the authors' original instantiation:
        self.seed = seed
        self.env.seed(self.seed)
        self.control_size = self.env.action_space.shape[0]
        self.control_bounds = [
            (self.env.action_space.low[i], self.env.action_space.high[i])
            for i in range(self.env.action_space.shape[0])
        ]
        self.time_steps = time_steps

        self.frame_delay_ms = frame_delay_ms

    def known_reset(self, seed: int = None):
        """Reset to a known starting-state.

        Converted from DemPref codebase.
        """
        if not seed:
            seed = self.seed
        self.env.seed(seed)
        state = self.env.reset()
        return state

    # def run(self, controls: np.ndarray, start_state: int) -> np.ndarray:
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
        obser = self.known_reset()
        # obser = self.known_reset(start_state)
        s = [obser]
        for i in range(self.time_steps):
            try:
                results = self.env.step(c[i])
            except:
                print("Caught unstable simulation; skipping.")
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
        start_state: int,  # start_state: np.ndarray,
        weights: np.ndarray,
        trajectory_length: int = 10,
    ) -> np.ndarray:
        def reward(controls, domain, weights):
            """
            One-step reward function.

            :x: state
            :weights: weight for reward function
            :return:
            """
            t = self.run(controls)
            feats = np.zeros((t[0].shape[0], self.w_dim))
            for i in range(len(t[0])):
                if i + 1 == t[0].shape[0]:
                    f = self.feature_fn(t[1][i], t[0][i], at_last_state=True)
                else:
                    f = self.feature_fn(t[1][i], t[0][i])
                feats[i, :] = f
            # Set the initial velocity equal to 0:
            feats[0, -2] = 0
            # Find the average value of the first three features:
            feats_final = np.mean(feats[:, :-1], axis=0)
            # Tack on the "distance from goal" feature unique to the
            # last state:
            feats_final = np.append(feats_final, feats[-1, -1])
            return -np.dot(weights, feats_final.T).squeeze()

        # weights = weights / np.linalg.norm(weights)
        # self.known_reset(seed)
        # self.env.reset()
        # self.state = start_state
        self.seed = start_state
        self.state = self.known_reset(start_state)

        low = [x[0] for x in self.control_bounds] * trajectory_length
        high = [x[1] for x in self.control_bounds] * trajectory_length
        optimal_ctrl = None
        opt_val = np.inf
        start = time.time()
        # 1.) Find the optimal controls given the start state and weights:
        for _ in range(self.optimal_trajectory_iters):
            temp_res = opt.fmin_l_bfgs_b(
                reward,
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
            if temp_res[1] < opt_val:
                optimal_ctrl = temp_res[0]
                opt_val = temp_res[1]
        end = time.time()
        print(
            "Finished generating optimal trajectory in "
            + str(end - start)
            + "s"
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
        current_time = time.perf_counter()
        # TODO Consider pickling:
        print("Saving trajectory ...")
        df.to_csv(
            self.output_path
            + Path(f"optimal_trajectory_{current_time:.0f}.csv")
        )

        # 2.) Extract the features from that optimal trajectory:
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
        # self.state = start_state
        return optimal_trajectory_final

    def watch(self, trajectory: np.ndarray, seed: int = None):
        # if len(t.controls[0][0]) == 1:
        if len(trajectory[1][0]) == 1:
            mapping = {1: [0, -1], 2: [1, 0], 3: [0, 1], 0: [0, 0]}
            controls = []
            for i in range(len(trajectory[1])):
                controls.append(mapping[trajectory[1][i][0]])
                # controls.append(mapping[t.controls[0][i][0]])
            # t = traj.Trajectory(t.states, np.array([controls]))
            trajectory = np.array(trajectory[1], np.array([controls]))

    def collect_demonstrations(
        self, num_dems: int = 1, world_lst: typing.List = None
    ) -> typing.List[str]:
        names = []
        for _ in range(num_dems):
            name = play(self.seed)
            names.append(name)
        return names

    def feature_fn(
        self, action, state, at_last_state: bool = False
    ) -> np.ndarray:  # traj.Trajectory):
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

        """

        def dist_from_landing_pad(state: np.ndarray):
            """Compute distance from the landing pad (which is at (0,0)).

            Note: weight should be negative.
            """
            return -15 * np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        def lander_angle(state: np.ndarray):
            """Compute lander's angle w.r.t. ground.

            Angle = 0 when lander is upright.

            Note: weight should be positive.
            """
            return 15 * np.exp(-np.abs(state[4]))

        def velocity(state: np.ndarray):
            """Compute the lander's velocity.

            Note: weight should be negative.
            """
            return -10 * np.exp(-np.sqrt(state[2] ** 2 + state[3] ** 2))

        def final_position(state: np.ndarray):
            """Lander's final state position.

            Note: the weight should be negative.
            """
            return -30 * np.exp(-np.sqrt(state[0] ** 2 + state[1] ** 2))

        # 1.) Compute the features of this new state:
        phi = np.stack(
            [
                dist_from_landing_pad(state),
                lander_angle(state),
                velocity(state),
                0,
            ]
        )
        # 2.) If we've reached the terminal state, compute the
        #     corresponding feature:
        if at_last_state:
            phi[-1] = final_position(state)
        return phi

    # def trajectory_feature_fn(
    #    self, trajectory: np.ndarray
    # ):  # traj.Trajectory):
    #    """Get the features from a trajectory.

    #    Note that DemPref's trajectories have multi-dimensional
    #    arrays--one for states and one for controls.

    #    [state] is of the form:
    #      [agent][time][states], where [states] comes from a gym.env.step()

    #    [control] is of the form:
    #      [agent][time][controls], where [controls] comes from some call
    #      to a gym.env.action_space.

    #    s (list): The state. Attributes:
    #              s[0] is the horizontal coordinate
    #              s[1] is the vertical coordinate
    #              s[2] is the horizontal speed
    #              s[3] is the vertical speed
    #              s[4] is the angle
    #              s[5] is the angular speed
    #              s[6] 1 if first leg has contact, else 0
    #              s[7] 1 if second leg has contact, else 0
    #    """

    #    # distance from landing pad at (0, 0)
    #    # weight should be negative
    #    def np_dist_from_landing_pad(x):
    #        return -15 * np.exp(-np.sqrt(x[0] ** 2 + x[1] ** 2))

    #    # angle of lander
    #    # angle is 0 when upright (positive in left direction,
    #    #   negative in right)
    #    # weight should be positive
    #    def np_lander_angle(x):
    #        return 15 * np.exp(-np.abs(x[4]))

    #    # velocity of lander
    #    # weight should be negative
    #    def np_velocity(x):
    #        return -10 * np.exp(-np.sqrt(x[2] ** 2 + x[3] ** 2))

    #    def np_path_length(t):
    #        """Compute trajectory path length.

    #        Note: weight should be positive.

    #        Despite its presence, the call to this function is stubbed out
    #        in DemPref codebase.
    #        """
    #        states = t.states[0]
    #        total = 0
    #        for i in range(1, len(states)):
    #            total += np.sqrt(
    #                (states[i][0] - states[i - 1][0]) ** 2
    #                + (states[i][1] - states[i - 1][1]) ** 2
    #            )
    #        total = np.exp(-total)
    #        return 10 * total

    #    # final position
    #    # weight should be negative
    #    def np_final_position(t):
    #        x = t.states[0][-1]
    #        return -30 * np.exp(-np.sqrt(x[0] ** 2 + x[1] ** 2))

    #    lst_of_features = []
    #    for i in range(len(t.states[0])):
    #        # Look at the 0th agent's states at time
    #        # i. Note there are 8 state-parameters for
    #        # lunar lander:
    #        x = t.states[0][i]
    #        if i > len(t.states) // 5:
    #            phi = np.stack(
    #                [
    #                    np_dist_from_landing_pad(x),
    #                    np_lander_angle(x),
    #                    np_velocity(x),
    #                ]
    #            )
    #        else:
    #            phi = np.stack(
    #                [np_dist_from_landing_pad(x), np_lander_angle(x), 0]
    #            )
    #        lst_of_features.append(phi)
    #    phi_total = list(np.mean(lst_of_features, axis=0))
    #    # phi_total.append(np_path_length(t))
    #    phi_total.append(np_final_position(t))
    #    return np.array(phi_total)
