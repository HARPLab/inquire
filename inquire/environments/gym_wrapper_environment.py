import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
from inquire.environments.environment import Environment


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
        optimal_trajectory_function: callable,
        output_path: str = None,
        actions_to_sample: int = 100,
    ):
        """Instantiate the environment.

        ::inputs:
            ::env_name: a title for the environment
            ::gym_compatible_env: a user-defined environment that has
                - a gym-compatible observation space
                - a gym-compatible action space
                - user-defined feature_function()
                - other user-specific functionality
            ::actions_to_sample: How many combinations of actions
                                 to generate when calling
                                 available_actions()
        """
        try:
            assert hasattr(gym_compatible_env, "observation_space")
            assert hasattr(gym_compatible_env, "action_space")
            self.env = gym_compatible_env
        except AssertionError:
            print(
                "User's gym environment %s is missing either a ",
                str(gym_compatible_env),
            )
            print("gym.env.observation_space or a gym.env.action_space.")
            return
        try:
            assert callable(optimal_trajectory_function)
        except AssertionError:
            print("User failed to define an optimal trajectory function.")
            return

        # Setup instance attributes:
        self.name = env_name.lower()
        self.rng = np.random.default_rng()
        self.actions_to_sample = actions_to_sample
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

    def __repr__(self) -> str:
        """Return the class' representative string."""
        return f"{self.__class__.__name__}"

    # Define the Environment class' virtual functions:
    def generate_random_state(self, random_state):
        """Generate random seed with which we reset env."""
        # When fed the same random_state, the environment returns to
        # the same reset state.
        rando = random_state.randint(low=0, high=sys.maxsize)
        return rando

    def generate_random_reward(self, random_state):
        """Randomly generate a weight vector for trajectory features."""
        random_reward = random_state.uniform(
            low=-1, high=1, size=(self.w_dim,)
        )
        random_reward = random_reward / np.linalg.norm(random_reward)
        return random_reward

    def optimal_trajectory_from_w(self, start_state: int, w: np.ndarray):
        """To be implemented within user-defined gym environment.

        Note: start_state must be a gym-compatible seed.
        """
        try:
            assert hasattr(self, "optimal_trajectory_fn")
            optimal_trajectory = self.optimal_trajectory_fn(start_state, w)
            return optimal_trajectory
        except AssertionError:
            print(
                "User's environment doesn't have an 'optimal_trajectory_fn' "
            )
            print("function. Returning.")
            return

    def features(
        self,
        action,
        state: Union[np.ndarray, int],
        trajectory: np.ndarray = None,
    ):
        """Extract features of trajectory from state after action."""
        try:
            assert hasattr(self, "feature_fn")
            if type(state) == int:
                # Received a seed; reset environment and get initial state:
                initial_state = self.reset(state)
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

            # Randomly sample an action for each actuator in the model:
            actions.append(
                np.random.uniform(
                    low=u[0][0],
                    high=u[1][0],
                    size=(self.env.action_space.shape),
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
            self.reset(current_state)
        # Apply the velocity for multiple timesteps:
        for a in range(self.timesteps_per_state):
            state, _, done, _ = self.env.step(action)
        return state

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
                        low=u[i][0], high=u[i][1], size=2000, endpoint=True
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
