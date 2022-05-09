"""A domain for real-world experiments."""
import time
from itertools import combinations
from pathlib import Path
from typing import Union

from inquire.environments.gym_wrapper_environment import Environment
from inquire.interactions.feedback import Trajectory

from numba import jit

import numpy as np

import pandas as pd


class Pizza(Environment):
    """Create a pepperoni pizza by placing toppings."""

    def init(
        self,
        seed: int = None,
        topping_max: int = 30,
        pizza_form: dict = None,
        basis_functions: list = None,
        basis_function_arguments: dict = None,
        output_path: str = str(Path.cwd()) + "/output/pizza_environment/",
        normalize: bool = False,
        verbose: bool = True,
    ):
        """Initialize a domain for creating pizzas.

        ::inputs:
            ::topping_max: The total number of toppings a pizza can have; akin
                           to a trajectory length.
            ::pizza_form: The high-level attributes of a pizza.
        """
        self._seed = seed
        self._normalize = normalize
        self._output_path = output_path
        self._rng = np.random.default_rng(self._seed)
        self._topping_max = topping_max
        self._pizza_form = pizza_form
        crust_and_topping_factor = (self._pizza_form["crust_thickness"]) + (
            self._pizza_form["topping_diam"] / 2.0
        )
        self._viable_surface_radius = (
            self._pizza_form.diameter / 2.0 - crust_and_topping_factor
        )
        self._viable_surface_area = np.pi * self._viable_surface_radius ** 2
        self._area_per_topping = (
            np.pi * (self._pizza_form["topping_diam"] / 2.0) ** 2
        )
        self._x_coordinate_range = np.linspace(
            -self._viable_surface_radius,
            self._viable_surface_radius,
            1000,
            endpoint=True,
        )
        self._y_coordinate_range = np.array(
            self._x_coordinate_range, copy=True
        )
        self._basis_fns = basis_functions
        self._basis_fn_arguments = basis_function_arguments
        basis_fn_memory_blocks = []
        # Make a composition of basis functions:
        for b in basis_functions:
            if b == "approximate_coverage":
                basis_fn_memory_blocks.append(
                    self.approximate_surface_coverage
                )
            elif b == "approximate_overlap_last_to_all":
                basis_fn_memory_blocks.append(self.approximate_overlap_last)
            elif b == "avg_magnitude_last_to_all":
                basis_fn_memory_blocks.append(self.avg_magnitude_last_to_all)
            elif b == "last_point_x_variance":
                basis_fn_memory_blocks.append(self.last_point_x_variance)
            elif b == "last_point_y_variance":
                basis_fn_memory_blocks.append(self.last_point_y_variance)
            elif b == "markovian_magnitude":
                basis_fn_memory_blocks.append(self.markovian_magnitude)

        # The param function is a composition of the basis functions:

        def param_fn(topping_coords: np.ndarray) -> np.ndarray:
            """Compute the parameters of a set of topping placements."""
            param_values = np.array([])
            # For each basis function in this param function:
            for i, b in enumerate(basis_fn_memory_blocks):
                # Compute the param value:
                param_values = np.append(param_values, b(topping_coords))
            return param_values

        self.compute_params = param_fn

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate a random starting state.

        ::inputs:
            ::random_state: A number generator object; instead use this
                            class' rng instance attribute.
        """
        pizza_topping_count = self._rng.randint(low=0, high=self._topping_max)
        # Generate pizza_topping_count x,y coordinates:
        topping_coordinates = self.generate_2D_points(
            radius=self._viable_surface_radius, count=pizza_topping_count
        )
        return topping_coordinates

    def generate_random_reward(self, random_state) -> int:
        """Generate a random vector of weights.

        ::inputs:
            ::random_state: A number generator object; instead use this
                            class' rng instance attribute.
        """
        weights = self._rng.random((1, self.w_dim))
        weights = weights / np.sum(weights)
        return weights

    def optimal_trajectory_from_w(
        self, start_state: np.ndarray, w: Union[list, np.ndarray]
    ) -> np.ndarray:
        """Generate the optimal trajectory for given weights w."""
        self._seed = start_state
        self.reset()
        pass

    def features(
        self, action: list, state: Union[list, np.ndarray]
    ) -> np.ndarray:
        """Compute the features of state reached by action.

        ::inputs:
            ::action: The (x,y) coordinate of the topping that was most
                      recently placed on the pizza.
            ::state: The (x, y) coordinates of all toppings.
        """
        # Copy to avoid mutating:
        coords = np.array(state, copy=True).reshape(-1, 1)
        coords_features = self.compute_params(coords)
        return coords_features

    def available_actions(self, current_state: np.ndarray) -> list:
        """Return the possible topping placements given current_state."""
        if self.is_terminal_state(current_state):
            return [None, None]
        else:
            return [self._x_coordinate_range, self._y_coordinate_range]

    def next_state(
        self, current_state: np.ndarray, action: list
    ) -> np.ndarray:
        """Generate state after transition from current_state via action."""
        next_topping = np.array(action, copy=True).reshape(-1, 1)
        new_state = np.append(current_state, next_topping, axis=1)
        return new_state

    def is_terminal_state(self, current_state: np.ndarray) -> bool:
        """Check if more toppings can be added."""
        if current_state.shape[1] == self._topping_max:
            return False
        else:
            return True

    def all_actions(self) -> list:
        """All possible topping placements."""
        return [self._x_coordinate_range, self._y_coordinate_range]

    def state_space_dim(self):
        """Observation space is continuous; return None."""
        return np.inf

    def state_space(self):
        """Observation space is continuous; return None."""
        return np.inf

    def state_index(self, state):
        """Observation space is continuous; return None."""
        return None

    """The proceeding code consists of helper functions specific to this
    domain. No part of the Inquire framework should need to be modified to
    accommodate these functions. Note that not all of the proceeding functions
    must be used.
    """

    def reset(self, seed: int = None) -> None:
        """Reset the environment according to seed."""
        if seed is None:
            self._rng = np.random.default_rng(self._seed)
        else:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.random((self._state_vector_size, 1))
        self._goal_state = self._rng.integers(
            low=0,
            high=self._trajectory_length,
            size=(self._state_vector_size, 1),
        )

    @jit(nopython=True)
    def generate_2D_points(self, radius: float, count: int) -> np.ndarray:
        """Uniformly sample points from a circle."""
        pts = np.empty((2, count))
        for i in range(count):
            r = radius * np.sqrt(np.random.uniform(0, 1))
            theta = 2 * np.pi * np.random.uniform(0, 1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts[:, i] = np.array([x, y])
        return pts

    @jit(nopython=True)
    def min_normalize(data: np.ndarray) -> np.ndarray:
        """Shift and scale the data."""
        # Get the min and max:
        d_min = data.min()
        d_max = data.max()

        # Get the range:
        d_range = d_max - d_min

        # Subtract the min from each data-point:
        shifted_data = data - d_min

        # Divide by the range:
        if d_range > 0:
            normed_data = shifted_data / d_range
        else:
            normed_data = data
        return normed_data

    def approximate_surface_coverage(
        self, state: Union[list, np.ndarray]
    ) -> float:
        """Compute the approximate surface-area coverage."""
        coords = np.array(state, copy=True).reshape(-1, 1)
        topping_diam = self._pizza_form["topping_diam"]
        # ID the toppings actually ON the viable surface via:
        # ||object - surface origin||_2:
        sq_sum = np.sqrt(coords[0, :] ** 2 + coords[1, :] ** 2)
        inds = np.argwhere(sq_sum <= self._viable_surface_radius)[:, 0]

        overlapping_area = 0
        # If there's more than one topping, approximate their overlap:
        if coords.shape[1] > 1:
            xy_combos = np.array(list(combinations(coords[:, inds].T, 2)))
            x1 = xy_combos[:, 0, 0]
            x2 = xy_combos[:, 1, 0]
            y1 = xy_combos[:, 0, 1]
            y2 = xy_combos[:, 1, 1]
            xs = (x1 - x2) ** 2
            ys = (y1 - y2) ** 2
            topping_dists = np.sqrt(xs + ys)
            # Avoid division-by-zero errors:
            topping_dists = np.where(
                topping_dists == 0, 0.00001, topping_dists
            )
            # Heuristically compute total overlap area:
            overlapping_area = np.where(
                topping_dists < topping_diam,
                self._area_per_topping * np.exp(-topping_dists),
                0,
            )
            overlapping_area = np.sum(overlapping_area)

        # Compute the coverage approximation:
        approx_absolute_coverage = (
            self._area_per_topping * inds.shape[0] - overlapping_area
        )
        coverage = approx_absolute_coverage / self._viable_surface_area
        return coverage

    def approximate_overlap_last(
        self, state: Union[list, np.ndarray]
    ) -> float:
        """Approximate area last topping overlaps others."""
        coords = np.array(state, copy=True).reshape(-1, 1)
        # If there's more than one topping, approximate last topping's
        # overlap with each of the others:
        overlapping_area = 0
        if coords.shape[1] > 1:
            # Vectorize the last topping for efficient operations:
            most_recent = (
                coords[:, -1].reshape(2, 1).repeat(coords.shape[1], axis=1)
            )
            dists = (most_recent - coords)[:, :-1]
            mags = np.sqrt((dists ** 2).sum(axis=0))
            # Avoid division-by-zero errors:
            topping_dists = np.where(mags == 0, 0.00001, mags)
            # Heuristically compute total overlap area:
            overlapping_area = np.where(
                topping_dists < self._topping_diam,
                self._area_per_topping * np.exp(-topping_dists),
                0,
            )
            overlapping_area = overlapping_area.sum()
        return overlapping_area

    def last_point_x_variance(
        self, state: Union[list, np.ndarray] = None
    ) -> float:
        """Compute x variance of last point.

        Upper bound of discrete, uniform random variable is (b-a)**2 / 4
        """
        coords = np.array(state, copy=True).reshape(-1, 1)
        if coords.shape[1] <= 1:
            return 0
        last_x = coords[0, -1].repeat(coords.shape[1]).reshape(1, -1)
        diffs = (last_x - coords[0, :])[0, :-1].reshape(1, -1)
        if diffs.shape[1] <= 1:
            var_x_to_last = np.sum(diffs ** 2)
        else:
            var_x_to_last = np.sum(diffs ** 2) / (diffs.shape[1] - 1)
        return var_x_to_last

    def last_point_y_variance(
        self, state: Union[list, np.ndarray] = None
    ) -> float:
        """Compute y variance of last point."""
        coords = np.array(state, copy=True).reshape(-1, 1)
        if coords.shape[1] <= 1:
            return 0
        last_y = coords[1, -1].repeat(coords.shape[1]).reshape(1, -1)
        diffs = (last_y - coords[1, :])[0, :-1].reshape(1, -1)
        if diffs.shape[1] <= 1:
            var_y_to_last = np.sum(diffs ** 2)
        else:
            var_y_to_last = np.sum(diffs ** 2) / (diffs.shape[1] - 1)
        return var_y_to_last

    def markovian_magnitude(
        self, state: Union[list, np.ndarray] = None
    ) -> float:
        """Compute magnitude between latest topping and one preceding it."""
        coords = np.array(state, copy=True).reshape(-1, 1)
        if coords.shape[1] <= 1:
            return 0
        x_dist = coords[0, -1] - coords[0, -2]
        y_dist = coords[1, -1] - coords[1, -2]
        mag = np.sqrt(x_dist ** 2 + y_dist ** 2)

        return mag

    def avg_magnitude_last_to_all(
        self, state: Union[list, np.ndarray] = None
    ) -> float:
        """Compute average magnitude between latest topping and all others."""
        coords = np.array(state, copy=True).reshape(-1, 1)
        if coords.shape[1] <= 1:
            return 0
        # Vectorize the last topping for efficient operations:
        most_recent = (
            coords[:, -1].reshape(2, 1).repeat(coords.shape[1], axis=1)
        )
        dists = (most_recent - coords)[:, :-1]
        mags = np.sqrt((dists ** 2).sum(axis=0))
        avg_mags = mags.sum() / mags.shape[0]
        return avg_mags

    def stepwise_pizza_generator(
        self,
        params: np.ndarray,
        learned_weights: np.ndarray,
        topping_samples: int = 5,
    ) -> object:
        """Incrementally add fixed amount of toppings.

        ::inputs:
          ::topping_samples: How many coords to sample for the next
                             topping's placement.
        """
        start = time.perf_counter()
        toppings = np.empty((2, 1))
        first_slice = True
        latest_reward = 0
        inner_reward = 0
        latest_toppings = None

        new_toppings = self.generate_2D_points(
            self._viable_surface_radius, topping_samples
        )
        if self._topping_max == 1:
            for i in range(new_toppings.shape[1]):
                temp_toppings = new_toppings[:, i].reshape(-1, 1)
                params_candidate = self.compute_params(temp_toppings).reshape(
                    1, -1
                )
                if params.shape[0] > 1:
                    features = self.features(params_candidate, params)
                    new_rewards = learned_weights.T.dot(features)
                    new_reward = new_rewards.max()
                else:
                    features = (
                        self.features(params_candidate, params)
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
                params_candidate = self.compute_params(temp_toppings).reshape(
                    1, -1
                )
                if params.shape[0] > 1:
                    features = self.features(params_candidate, params)
                    new_rewards = learned_weights.T.dot(features)
                    new_reward = new_rewards.max()
                else:
                    features = (
                        self.features(params_candidate, params)
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
        elapsed = time.perf_counter() - start
        if self._verbose:
            print(f"It took {elapsed:.3f} seconds to generate a pizza")
        return toppings
