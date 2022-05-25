"""A domain for real-world experiments."""
import time
from itertools import combinations
from pathlib import Path
from typing import Tuple, Union

from inquire.environments.gym_wrapper_environment import Environment
from inquire.interactions.feedback import Trajectory

from numba import jit

import numpy as np

import pandas as pd


class Pizza(Environment):
    """Create a pepperoni pizza by placing toppings."""

    def __init__(
        self,
        seed: int = None,
        max_topping_count: int = 25,
        topping_sample_count: int = 1500,
        optimization_iteration_count: int = 300,
        pizza_form: dict = None,
        basis_functions: list = None,
        output_path: str = str(Path.cwd()) + "/output/pizza_environment/",
        verbose: bool = False,
    ):
        """Initialize a domain for creating pizzas.

        ::inputs:
            ::topping_max: The total number of toppings a pizza can have; akin
                           to a trajectory length.
            ::pizza_form: The high-level attributes of a pizza.
        """
        self._seed = seed
        self._max_topping_count = max_topping_count
        self._topping_sample_count = topping_sample_count
        self._pizza_form = pizza_form
        self._optimization_iteration_count = optimization_iteration_count
        self._output_path = output_path
        self._verbose = verbose

        self._rng = np.random.default_rng(self._seed)
        crust_and_topping_factor = (self._pizza_form["crust_thickness"]) + (
            self._pizza_form["topping_diam"] / 2.0
        )
        self._viable_surface_radius = (
            self._pizza_form["diameter"] / 2.0 - crust_and_topping_factor
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

        self.w_dim = len(basis_functions)
        self._basis_functions = basis_functions
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

        # The feature function is a composition of the basis functions:

        def feature_fn(topping_coords: np.ndarray) -> np.ndarray:
            """Compute the features of a set of topping placements."""
            feature_values = np.array([])
            # For each basis function in this feature function:
            for i, b in enumerate(basis_fn_memory_blocks):
                # Compute the feature value:
                feature_values = np.append(feature_values, b(topping_coords))
            return feature_values

        self.compute_features = feature_fn

    @property
    def basis_functions(self) -> dict:
        """Return dictionary of high-level attributes."""
        return self._basis_functions.copy()

    @property
    def pizza_form(self) -> dict:
        """Return dictionary of high-level attributes."""
        return self._pizza_form.copy()

    @property
    def viable_surface_radius(self) -> float:
        """Return radius of surface upon which toppings can be placed."""
        return self._viable_surface_radius

    @property
    def max_topping_count(self) -> int:
        """Return the maximum number of toppings to include on any pizza."""
        return self._max_topping_count

    def generate_random_state(self, random_state) -> np.ndarray:
        """Generate a random set of toppings to put on a pizza.

        ::inputs:
            ::random_state: A number generator object; instead use
                            self._rng
        """
        pizza_topping_count = self._rng.integers(
            low=1, high=self._max_topping_count - 1
        )
        # Generate pizza_topping_count x,y coordinates:
        topping_coordinates = generate_2D_points(
            radius=self._viable_surface_radius, count=pizza_topping_count
        )
        return topping_coordinates

    def generate_random_reward(self, random_state) -> int:
        """Generate a random vector of weights.

        ::inputs:
            ::random_state: A number generator object; instead use this
                            class' rng instance attribute.
        """
        weights = self._rng.random((self.w_dim,))
        for i in range(weights.shape[0]):
            sign = self._rng.choice((-1, 1), 1)
            weights[i] = sign * weights[i]
        weights = weights / np.linalg.norm(weights)
        # weights = weights / np.sum(weights)
        return weights.squeeze()

    def optimal_trajectory_from_w(
        self, start_state: np.ndarray, w: Union[list, np.ndarray]
    ) -> np.ndarray:
        """Find placement of next topping which yields greatest reward.

        ::inputs:
            ::start_state: A set of (x,y) coordinates of topping placements.
            ::w: A vector of weights corresponding to the domain's features.

        """
        start = time.perf_counter()
        toppings = np.array(start_state, copy=True)
        best_reward = None
        best_toppings = None
        best_features = None
        for i in range(self._optimization_iteration_count):
            # Generate a bunch of potential positions for the next slice:
            new_toppings = generate_2D_points(
                self._viable_surface_radius, self._topping_sample_count
            )
            # See which of new_toppings yields the greatest reward:
            for j in range(new_toppings.shape[1]):
                temp_toppings = np.hstack(
                    (toppings, new_toppings[:, j].reshape(-1, 1))
                )
                temp_features = self.features(
                    new_toppings[:, j], temp_toppings
                ).squeeze()
                new_reward = w.dot(temp_features.T)
                if best_reward is None:
                    best_reward = new_reward
                if new_reward > best_reward:
                    best_reward = new_reward
                    best_toppings = np.array(temp_toppings, copy=True)
                    best_features = np.array(temp_features, copy=True)
        best_topping_placements = Trajectory(best_toppings, best_features)
        if self._verbose:
            elapsed = time.perf_counter() - start
            print(
                f"It took {elapsed:.2} seconds to find next topping placement."
            )
        return best_topping_placements

    def features(
        self, action: Union[np.ndarray, list], state: Union[list, np.ndarray]
    ) -> np.ndarray:
        """Compute the features of state reached by action.

        ::inputs:
            ::action: The (x,y) coordinate of the topping that was most
                      recently placed on the pizza.
            ::state: The (x, y) coordinates of all toppings. Shape: 2-by-n.
        """
        # If there are no toppings or just one topping, we have no features
        # to compute:
        if state.shape[1] == 1:
            return np.zeros((self.w_dim,))
        else:
            # Copy to avoid mutating:
            coords = np.array(state, copy=True)
            coords_features = self.compute_features(coords)
            return coords_features.squeeze()

    def available_actions(self, current_state: np.ndarray) -> list:
        """Return the possible topping placements given current_state."""
        if self.is_terminal_state(current_state):
            return []
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
        if current_state.shape[1] >= self._max_topping_count:
            return True
        else:
            return False

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

    """

    The proceeding code consists of domain-specific helper functions. No part
    of the Inquire framework should need to be modified to accommodate these
    functions.

    """

    def approximate_surface_coverage(
        self, state: Union[list, np.ndarray]
    ) -> float:
        """Compute the approximate surface-area coverage."""
        coords = np.array(state, copy=True)
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
        coords = np.array(state, copy=True)
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
                topping_dists < self._pizza_form["topping_diam"],
                self._area_per_topping * np.exp(-topping_dists),
                0,
            )
            overlapping_area = overlapping_area.sum()
        # return overlapping_area
        return -np.exp(-overlapping_area)

    def last_point_x_variance(self, state: Union[list, np.ndarray]) -> float:
        """Compute x variance of last point.

        Upper bound of discrete, uniform random variable is (b-a)**2 / 4
        """
        coords = np.array(state, copy=True)
        if coords.shape[1] <= 1:
            return 0
        last_x = coords[0, -1].repeat(coords.shape[1]).reshape(1, -1)
        diffs = (last_x - coords[0, :])[0, :-1].reshape(1, -1)
        if diffs.shape[1] <= 1:
            var_x_to_last = np.sum(diffs ** 2)
        else:
            var_x_to_last = np.sum(diffs ** 2) / (diffs.shape[1] - 1)
        return var_x_to_last

    def last_point_y_variance(self, state: Union[list, np.ndarray]) -> float:
        """Compute y variance of last point."""
        coords = np.array(state, copy=True)
        if coords.shape[1] <= 1:
            return 0
        last_y = coords[1, -1].repeat(coords.shape[1]).reshape(1, -1)
        diffs = (last_y - coords[1, :])[0, :-1].reshape(1, -1)
        if diffs.shape[1] <= 1:
            var_y_to_last = np.sum(diffs ** 2)
        else:
            var_y_to_last = np.sum(diffs ** 2) / (diffs.shape[1] - 1)
        return var_y_to_last

    def markovian_magnitude(self, state: Union[list, np.ndarray]) -> float:
        """Compute magnitude between latest topping and one preceding it."""
        coords = np.array(state, copy=True)
        if coords.shape[1] <= 1:
            return 0
        x_dist = coords[0, -1] - coords[0, -2]
        y_dist = coords[1, -1] - coords[1, -2]
        mag = np.sqrt(x_dist ** 2 + y_dist ** 2)

        return mag

    def avg_magnitude_last_to_all(
        self, state: Union[list, np.ndarray]
    ) -> float:
        """Compute average magnitude between latest topping and all others."""
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        # Vectorize the last topping for efficient operations:
        most_recent = (
            coords[:, -1].reshape(2, 1).repeat(coords.shape[1], axis=1)
        )
        # Compute the distances; ignore the last distance which is the distance
        # from itself:
        dists = (most_recent - coords)[:, :-1]
        mags = np.sqrt((dists ** 2).sum(axis=0))
        avg_mag = mags.sum() / mags.shape[0]
        return avg_mag


"""

The proceeding are Numba-optimized helper functions. No part of the Inquire
framework should need to be modified to accommodate these functions.

"""


@jit(nopython=True)
def generate_2D_points(radius: float, count: int) -> np.ndarray:
    """Uniformly sample points from a circle."""
    pts = np.empty((2, count))
    for i in range(count):
        r = radius * np.sqrt(np.random.uniform(0, 1))
        theta = 2 * np.pi * np.random.uniform(0, 1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts[:, i] = np.array([x, y])
    return pts
