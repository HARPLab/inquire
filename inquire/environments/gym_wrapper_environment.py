from inquire.environments.environment import Environment

import gym
import sys
import logging
from logging import handlers
import time
import numpy as np
from pathlib import Path


class GymWrapperEnvironment(Environment):
    """A wrapper class for OpenAI Gym.

    In theory, this class is provided a user-defined environment
    that ALREADY is compatible with gym. This wrapper then extracts
    the Environment functions from the standard gym.Env functionality
    such that it is ALSO compatible with our repository.
    """

    def __init__(
        self,
        env_name: str,
        gym_compatible_env,
        optimal_traj_function: callable,
        output_path: str = None,
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
        try:
            assert hasattr(gym_compatible_env, "observation_space")
            assert hasattr(gym_compatible_env, "action_space")
            self.env = gym_compatible_env
        except AssertionError:
            print(
                "User's gym environment %s is missing either/or a ",
                str(gym_compatible_env),
            )
            print("gym.env.observation_space or a gym.env.action_space.")
            return
        try:
            assert callable(optimal_traj_function)
        except AssertionError:
            print("User failed to define an optimal trajectory function.")
            return

        # Setup instance attributes:
        self.name = env_name.lower()
        self.rng = np.random.default_rng()
        self.done = False

        if output_path:
            if Path(output_path).exists():
                self.output_path = output_path
            else:
                self.output_path = Path(output_path).mkdir(parents=True)
        else:
            self.output_path = str(
                Path.cwd() / Path("/tests/output/" + self.name)
            )
            Path(self.output_path).mkdir()

    # Define the Environment class' virtual functions:
    def generate_random_state(self, random_state):
        """Generate random seed with which we reset env."""
        # When fed the same random_state, the environment returns to
        # the same reset state.
        rando = random_state.randint(low=0, high=sys.maxsize)
        return rando

    def generate_random_reward(self, random_state):
        """Randomly generate a weight vector for trajectory features.

        Note that for lunar lander's features, random weights should
        have the signs:
            [negative, positive, negative, negative]
        """
        # TODO Generalize this function

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
            print("User's environment doesn't have an optimal_trajectory() ")
            print("function. Returning.")
            return

    def features(self, action, state, trajectory: np.ndarray = None):
        """Extract features of trajectory from state after action."""
        try:
            assert hasattr(self, "feature_fn")
            if type(state) == int:
                # We got a seed; reset environment and get initial state:
                initial_state = self.known_reset(state)
                feats = self.feature_fn(action, initial_state)
            else:
                feats = self.feature_fn(action, state)
            return feats
        except AssertionError:
            print("User's environment doesn't have a feature_fn(). ")
            print("Can't extract features. Returning.")
            return

    def available_actions(self, current_state):
        """Return domain's range (continuous) or set (discrete) of actions."""
        if "box" in str(type(self.env.action_space)).split("."):
            # .action_space.shape reveals the number of actuators in the model.
            # .action_space.high/low reveals the control-value bounds of
            # those actuators
            u = [self.env.action_space.low, self.env.action_space.high]
            actions = []

            # For each actuator in the model, discretize the (low, high)
            # control ranges:

            # TODO Determine if discretization granularity is appropriate:

            for i in range(self.env.action_space.shape[0]):
                actions.append(
                    np.linspace(
                        start=u[0][0], stop=u[1][0], num=2000, endpoint=True
                    )
                )
        else:
            # The action space is discrete; return that set of actions:
            actions = np.ndarray.tolist(np.arange(self.env.action_space.n))
        return actions

    def next_state(self, current_state, action):
        """Return the next state after action."""
        if type(current_state) == int:
            # Reset the environment from seed:
            _ = self.known_reset(current_state)
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
            # .action_space.high/low reveals the control-value bounds of
            # those actuators
            u = [self.env.action_space.low, self.env.action_space.high]
            actions = []
            # For each actuator in the model, discretize the (low, high)
            # control ranges:

            # TODO Determine if this discretization granularity is appropriate:

            for i in range(self.env.action_space.shape[0]):
                actions.append(
                    np.linspace(
                        low=u[i][0], high=u[i][1], size=10000, endpoint=True
                    )
                )
        else:
            # The action space is discrete; return that set of actions:
            actions = np.ndarray.tolist(np.arange(self.env.action_space.n))
        return actions

    def state_space_dim(self):
        """Return dimensionality of the observation space."""
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
        """Return the state space."""
        # If the observation space is continuous, return the space's
        # dimensionality:
        if "box" in str(type(self.env.observation_space)).split("."):
            self.state_space_dim
        # Otherwise, return the discretized state-space representation:
        else:
            state_space = self.env.desc
            return state_space

    def state_index(self, state):
        """Return the index if the observation space is discrete."""
        if "box" in str(type(self.env.observation_space)).split("."):
            print(
                "Box environments are continuous; they "
                "don't have state indices. Returning None."
            )
            return None
        else:
            # The observation space is discrete; it is indexed in row-major
            # format according to gym's state-integer representation:
            return state
