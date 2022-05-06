"""A domain for real-world experiments."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from inquire.environments.gym_wrapper_environment import Environment
from inquire.interactions.feedback import Trajectory
from numba import jit


class Pizza(Environment):
    """Create a pepperoni pizza by placing toppings."""

    def init(
        self,
        seed: int = None,
        topping_max: int = 30,
        pizza_form: dict = None,
        output_path: str = str(Path.cwd()) + "/output/pizza_environment/",
        verbose: bool = True,
    ):
        """Initialize a domain for creating pizzas.

        ::inputs:
            ::topping_max: The total number of toppings a pizza can have; akin
                           to a trajectory length.
            ::pizza_form: The high-level attributes of a pizza.
        """
        self._seed = seed
        self._output_path = output_path
        self._rng = np.random.default_rng(self._seed)
        self._topping_max = topping_max
        self._pizza_form = pizza_form
        crust_and_topping_factor = (self._pizza_form.crust_thickness) + (
            self._pizza_form.topping_size / 2.0
        )
        self._query_space = (
            self._pizza_form.diameter / 2.0 - crust_and_topping_factor
        )

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate a random starting state."""
        start = time.perf_counter()
        pizza_topping_count = self._rng.randint(low=2, high=self._topping_max)
        # Generate pizza_topping_count x,y coordinates:
        topping_coordinates = self.generate_2D_points(
            radius=self._query_space, count=pizza_topping_count
        )
        stop = time.perf_counter()
        elapsed = stop - start
        if self._verbose:
            print(f"It took {elapsed:.3f} seconds to generate a random state.")
        return topping_coordinates

    def generate_random_reward(self, random_state) -> int:
        """Generate a random vector of weights."""
        weights = self._rng.random((1, self.w_dim))
        weights = weights / np.sum(weights)
        return weights

    def optimal_trajectory_from_w(self, start_state, w) -> np.ndarray:
        """Generate the optimal trajectory for given weights."""
        self._seed = start_state
        self.reset()
        pass

    def features(self, action, state) -> np.ndarray:
        """Generate the features of a pizza."""
        pass

    def available_actions(self, current_state) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    def next_state(self, current_state, action) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    def is_terminal_state(self, current_state) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    def all_actions(self) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    def state_space_dim(self) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    def state_space(self) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    def state_index(self, state) -> np.ndarray:
        """Generate a random vector of weights."""
        pass

    @jit(nopython=True)
    def generate_2D_points(
        self, radius: float, count: int, rand_generator=None
    ) -> np.ndarray:
        """Uniformly sample points from a circle."""
        pts = np.empty((2, count))
        if rand_generator is None:
            for i in range(count):
                r = radius * np.sqrt(np.random.uniform(0, 1))
                theta = 2 * np.pi * np.random.uniform(0, 1)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts[:, i] = np.array([x, y])
        else:
            for i in range(count):
                r = radius * np.sqrt(rand_generator.uniform(0, 1))
                theta = 2 * np.pi * rand_generator.uniform(0, 1)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts[:, i] = np.array([x, y])
        return pts

    def reset(self) -> None:
        """Reset pertinent state attributes for new experiment."""
        pass

    def stepwise_pizza_generator(
        self,
        learned_params: np.ndarray,
        learned_weights: np.ndarray,
        param_function: callable,
        topping_samples: int = 5,
    ) -> object:
        """Incrementally add fixed amount of toppings.

        ::inputs:
          ::param_functions: Function which computes a pizza's parameter
                             values.
          ::topping_samples: How many positions to sample for the next
                             topping's placement.
        """
        toppings = np.empty((2, 1))
        first_slice = True
        latest_reward = 0
        inner_reward = 0
        latest_toppings = None

        new_toppings = self.generate_2D_points(
            self._query_space, topping_samples
        )
        if self._topping_max == 1:
            for i in range(new_toppings.shape[1]):
                temp_toppings = new_toppings[:, i].reshape(-1, 1)
                params_candidate = param_function.compute_params(
                    temp_toppings
                ).reshape(1, -1)
                if learned_params.shape[0] > 1:
                    features = self.features(params_candidate, learned_params)
                    new_rewards = learned_weights.T.dot(features)
                    new_reward = new_rewards.max()
                else:
                    features = (
                        self.features(params_candidate, learned_params)
                        .squeeze()
                        .reshape(1, -1)
                    )
                    new_reward = learned_weights.dot(features.T)
                if new_reward >= latest_reward:
                    latest_reward = new_reward
                    latest_toppings = temp_toppings
            return latest_toppings

        while toppings.shape[1] < self._topping_max:
            for i in range(new_toppings.shape[1]):
                if first_slice:
                    temp_toppings = new_toppings[:, i].reshape(-1, 1)
                else:
                    temp_toppings = np.hstack(
                        (toppings, new_toppings[:, i].reshape(-1, 1))
                    )
                params_candidate = param_function.compute_params(
                    temp_toppings
                ).reshape(1, -1)
                if learned_params.shape[0] > 1:
                    features = self.features(params_candidate, learned_params)
                    new_rewards = learned_weights.T.dot(features)
                    new_reward = new_rewards.max()
                else:
                    features = (
                        self.features(params_candidate, learned_params)
                        .squeeze()
                        .reshape(1, -1)
                    )
                    new_reward = learned_weights.dot(features.T)
                if new_reward >= latest_reward:
                    latest_reward = new_reward
                    toppings = np.array(temp_toppings, copy=True)
                    topping_index = i
                    inner_reward = 0
                    break
                elif new_reward > inner_reward:
                    inner_reward = new_reward
                    inner_toppings = np.array(temp_toppings, copy=True)
                    topping_index = i

            if (i + 1) == new_toppings.shape[1]:
                toppings = np.array(inner_toppings, copy=True)
                latest_reward = inner_reward
                inner_reward = 0
            new_toppings = np.delete(new_toppings, topping_index, axis=1)
            if toppings.shape[1] == 1:
                first_slice = False
        return toppings
