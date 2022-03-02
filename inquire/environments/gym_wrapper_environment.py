from inquire.environments.environment import Environment

import gym
import sys
import logging
from logging import handlers
import time
import numpy as np
from pathlib import Path

# path = Path.cwd()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# formatted_output = "{asctime}|{name}|{levelname}|{message}"
# formatter_1 = logging.Formatter(formatted_output, style="{")
#
# handler_1 = logging.StreamHandler()
# handler_1.setLevel(logging.INFO)
# handler_1.setFormatter(formatter_1)
#
# handler_2 = handlers.RotatingFileHandler(path + Path("/data/gym_wrapper.log"))
# handler_2.setLevel(logging.DEBUG)
# handler_2.setFormatter(formatter_1)
#
# logger.addHandler(handler_1)
# logger.addHandler(handler_2)


class GymWrapperEnvironment(Environment):
    """A wrapper class for OpenAI Gym.

    In theory, this class is provided a user-defined environment
    that ALREADY is compatible with gym. This wrapper then extracts
    the Environment functions from the standard gym.Env functionality
    such that it is ALSO compatible with our repository.
    """

    def __init__(
        self, env_name: str, gym_compatible_env, optimal_traj_function
    ):
        """Instantiate the environment.

        ::inputs:
            ::env_name: a title for the environment
            ::gym_compatible_env: a user-defined environment that has
                - a gym-compatible observation space
                - a gym-compatible action space
                - user-defined feature_function()
                - other user-specific functionality
        """
        self.name = env_name.lower()
        self.rng = np.random.default_rng(
            40
        )  # 40 is an arbitrarily-chosen seed
        self.done = False

        # If env_name not in registry, make env and then
        # pass to Wrapper:
        try:
            assert hasattr(gym_compatible_env, "observation_space")
            assert hasattr(gym_compatible_env, "action_space")
            self.env = gym_compatible_env
        except AssertionError:
            print(
                "User's gym environment %s is missing either/or a gym.env.observation_space or a gym.env.action_space.",
                str(gym_compatible_env),
            )
            return
        try:
            assert callable(optimal_traj_function)
            # self.optimal_trajectory_from_w = optimal_traj_function
        except AssertionError:
            print("User failed to define an optimal trajectory function.")
            return

    # Now define the Environment class' virtual functions:
    def generate_random_state(self, random_state):
        """Generate random seed with which we reset env."""
        # (ii) When fed the same random_state, the environment will return to
        # the same reset state.
        rando = random_state.randint(low=0, high=sys.maxsize)
        # self.env.seed(rando)
        # return self.env.reset()
        return rando

    def generate_random_reward(self, random_state):
        """Randomly generate a weight vector for trajectory features.

        Note that for the four features, random weights should
        still have the signs:
            [negative, positive, negative, negative]
        """
        # random_reward = np.random.rand(1, self.w_dim)
        # for i in range(random_reward.shape[1]):
        #    sign = np.random.choice((-1, 1), 1)
        #    random_reward[0, i] = sign * random_reward[0, i]
        # random_reward = random_reward / np.linalg.norm(random_reward)

        # The ground-truth weights according to DemPref paper:
        random_reward = np.array([-0.4, 0.4, -0.2, -0.7])
        return random_reward

    def optimal_trajectory_from_w(self, start_state, w):
        """To be implemented within user-defined gym environment.

        Note that start_state should be a gym-compatible seed.
        """
        try:
            assert hasattr(self, "optimal_trajectory")
            optimal_trajectory_from_w = self.optimal_trajectory(start_state, w)
            return optimal_trajectory_from_w
        except AssertionError:
            print(
                "User's environment doesn't have an optimal_trajectory() function. Returning."
            )
            return

    def features(self, action, state, trajectory: np.ndarray = None):
        """Extract features of trajectory from state after action."""
        try:
            assert hasattr(self, "feature_fn")
            if type(state) == int:
                # We've been fed a seed; reset the environment and get initial
                # state:
                initial_state = self.known_reset(state)
                feats = self.feature_fn(action, initial_state)
            else:
                feats = self.feature_fn(action, state)
            return feats
        except AssertionError:
            print(
                "User's environment doesn't have a feature_fn(). Can't extract features."
            )
            return

    def available_actions(self, current_state):
        if "box" in str(type(self.env.action_space)).split("."):
            # .action_space.shape reveals the number of actuators in the model.
            # .action_space.high/low reveals the control-value bounds of those actuators
            u = [self.env.action_space.low, self.env.action_space.high]
            actions = []
            # For each actuator in the model, discretize the (low, high) control ranges:
            for i in range(self.env.action_space.shape[0]):
                actions.append(
                    np.linspace(
                        start=u[i][0], stop=u[i][1], num=10000, endpoint=True
                    )
                )
        else:
            # The action space is discrete; return that range:
            actions = np.ndarray.tolist(np.arange(self.env.action_space.n))
        return actions

    def next_state(self, current_state, action):
        """Return the next environment state after action."""
        if type(current_state) == int:
            # We've been fed a seed; reset the environment and proceed:
            curr_state = self.known_reset(current_state)
            self.state, _, self.done, _ = self.env.step(action)
        else:
            self.state, _, self.done, _ = self.env.step(action)
        return self.state

    def is_terminal_state(self, current_state):
        """Determine if in the final state."""
        if self.done:
            return True
        else:
            return False

    def all_actions(self):
        """Get all action commands from the environment."""
        if "box" in str(type(self.env.action_space)).split("."):
            # .action_space.shape reveals the number of actuators in the model.
            # .action_space.high/low reveals the control-value bounds of those actuators
            u = [self.env.action_space.low, self.env.action_space.high]
            actions = []
            # For each actuator in the model, discretize the (low, high) control ranges:
            for i in range(self.env.action_space.shape[0]):
                actions.append(
                    np.linspace(
                        low=u[i][0], high=u[i][1], size=10000, endpoint=True
                    )
                )
        else:
            # The action space is discrete; return that range:
            actions = np.ndarray.tolist(np.arange(self.env.action_space.n))
        return actions

    def state_space_dim(self):
        # NOTE is this w.r.t. the physical space alone, or is it w.r.t.
        #     the possible state attributes of the agent? (e.g. velocity)
        if "box" in str(type(self.env.observation_space)).split("."):
            if self.env.observation_space.is_bounded():
                dimensionality = self.env.observation_space.shape
                return dimensionality
            else:
                # The space is unbounded:
                return np.inf
        else:
            dimensionality = self.env.observation_space.shape
            return dimensionality

    def state_space(self):
        # NOTE is this w.r.t. the physical space alone, or is it w.r.t.
        #     the possible state attributes of the agent? (e.g. velocity)
        # NOTE how is this different from state_space_dim?
        if "box" in str(type(self.env.observation_space)).split("."):
            pass
        else:
            # Observation-space is discrete; return its description:
            return self.env.desc

    def state_index(self, state):
        """Return the index if the observation space is discrete."""
        if "box" in str(type(self.env.observation_space)).split("."):
            print(
                "Box environments are continuous; they "
                "don't have state indices. Returning None."
            )
            return
        else:
            # The observation space is discrete; it is indexed in row-major
            # format according to the state-integer representation:
            return state
