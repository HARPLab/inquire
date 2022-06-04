"""A domain for real-world experiments."""
import time
from itertools import combinations
from pathlib import Path
from typing import Union

from inquire.environments.gym_wrapper_environment import Environment
from inquire.interactions.feedback import Trajectory

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from numba import jit

import numpy as np

import pandas as pd


class PizzaMaking(Environment):
    """Create a pepperoni pizza by placing toppings."""

    def __init__(
        self,
        seed: int = None,
        max_topping_count: int = 20,
        topping_sample_count: int = 1500,
        optimization_iteration_count: int = 300,
        pizza_form: dict = None,
        basis_functions: list = None,
        output_path: str = str(Path.cwd()) + "/output/pizza_environment/",
        verbose: bool = False,
        debug: bool = True,
    ):
        """Initialize a domain for creating pizzas.

        ::inputs:
            ::topping_max: The total number of toppings a pizza can have; akin
                           to a trajectory length.
            ::pizza_form: The high-level attributes of a pizza.
        """
        self._seed = seed
        self._debug = debug
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

        self._feature_count = len(basis_functions)
        self._compare_to_desired = True
        self._desired_params = {
            "markovian_direction": 0.0,
            "markovian_magnitude": self._pizza_form["topping_diam"],
        }

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
            elif b == "dist_0_quadratic":
                basis_fn_memory_blocks.append(self.dist_0_quadratic)
            elif b == "dist_2_quadratic":
                basis_fn_memory_blocks.append(self.dist_2_quadratic)
            elif b == "dist_4_quadratic":
                basis_fn_memory_blocks.append(self.dist_4_quadratic)
            elif b == "distance_to_nearest_neighbor":
                basis_fn_memory_blocks.append(
                    self.distance_to_nearest_neighbor
                )
            elif b == "last_point_x_variance":
                basis_fn_memory_blocks.append(self.last_point_x_variance)
            elif b == "last_point_y_variance":
                basis_fn_memory_blocks.append(self.last_point_y_variance)
            elif b == "markovian_direction":
                basis_fn_memory_blocks.append(self.markovian_direction)
            elif b == "markovian_magnitude":
                basis_fn_memory_blocks.append(self.markovian_magnitude)
            elif b == "x_coordinate":
                basis_fn_memory_blocks.append(self.x_coordinate)
            elif b == "y_coordinate":
                basis_fn_memory_blocks.append(self.y_coordinate)

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
        if self._debug:
            # Assume the features to be markovian_direction and
            # markovian_magnitude when testing:
            generated = np.array([-2.99, -0.01])
        else:
            generated = self._rng.uniform(
                low=-5, high=5, size=(self._feature_count,)
            )
        generated = generated / np.linalg.norm(generated)
        print(f"GT reward: {generated}.")
        return generated

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
                f"It took {elapsed:.2f}s to find the best placement for the "
                "next topping."
            )
            print(
                f"That placement yielded a reward of {best_reward} and had "
                f"feature values: {best_features}."
            )
        return best_topping_placements

    def action_space(self) -> np.ndarray:
        """Get the environment's action space."""
        pass

    def trajectory_rollout(self, start_state, actions):
        pass

    def features_from_trajectory(self, trajectory):
        pass

    def w_dim(self) -> np.ndarray:
        """Return the dimensionality environment's features."""
        return self._feature_count

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
            return np.zeros((self._feature_count,))
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
            next_topping = self._rng.choice(
                self._x_coordinate_range, 2
            ).reshape(-1, 1)
            return [next_topping]
        # return [self._x_coordinate_range, self._y_coordinate_range]

    def next_state(
        self, current_state: np.ndarray, action: list
    ) -> np.ndarray:
        """Generate state after transition from current_state via action."""
        # next_topping = self._rng.choice(self._x_coordinate_range, 2).reshape(
        #    -1, 1
        # )
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

    def trajectory_from_states(self, states, features):
        return Trajectory(states, np.sum(features, axis=0))

    def distance_between_trajectories(self, a, b):
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
        """Approximate area last topping overlaps all other ."""
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
        return overlapping_area
        # return -np.exp(-overlapping_area)

    def last_point_x_variance(self, state: Union[list, np.ndarray]) -> float:
        """Compute x variance of last point.

        Upper bound on variance of discrete, uniform random variable is
        (b-a)**2 / 4
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
        """Compute y variance of last point.

        Upper bound on variance of discrete, uniform random variable is
        (b-a)**2 / 4
        """
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
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
        """Compute magnitude between latest topping and its precedecessor."""
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        x_dist = coords[0, -1] - coords[0, -2]
        y_dist = coords[1, -1] - coords[1, -2]
        mag = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if self._compare_to_desired:
            normed_distance = np.abs(
                mag - self._desired_params["markovian_magnitude"]
            ) / (self._viable_surface_radius * 2)
            # print(f"normed distance: {normed_distance}.")
            # print(f"expo'd normed distance: {10 * np.exp(-normed_distance)}.")
            return normed_distance
        # return 10 * np.exp(
        # -normed_distance
        # )
        else:
            return mag

    def markovian_direction(self, state: Union[list, np.ndarray]) -> float:
        """Compute direction to the latest topping from its predecessor.

        Up ~ 90 degrees
        Down ~ -90 degrees
        Left ~ +/-180 degrees
        Right ~ 0 degrees
        """
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0

        topping_a = coords[:, -2].reshape(-1, 1)
        topping_b = coords[:, -1].reshape(-1, 1)

        # Assume topping_a is the reference and shift the vector to the origin:
        transformed_b = topping_b - topping_a

        # Compute the direction from topping_a to topping_b:
        direction_in_rads = np.arctan2(transformed_b[1], transformed_b[0])
        direction_in_degrees = direction_in_rads * 180.0 / np.pi

        if self._compare_to_desired:
            desired = self._desired_params["markovian_direction"]
            if np.sign(direction_in_degrees) != np.sign(desired):
                # Convert negative value to positive value in 360 frame:
                neg_value = min(direction_in_degrees, desired)
                pos_value = max(direction_in_degrees, desired)
                neg_value_360 = 360 + neg_value
                # Compute the difference between them:
                diff = np.abs(pos_value - neg_value_360)
                # Deduce the actual magnitude of that difference:
                from_0 = np.abs(0 - diff)
                from_360 = np.abs(360 - diff)
                normed_final_diff = min(from_0, from_360) / 360.0
                return normed_final_diff
            # return 10 * np.exp(-normed_final_diff)
            else:
                return np.abs(direction_in_degrees - desired)
        else:
            return direction_in_degrees

    def x_coordinate(self, state: Union[list, np.ndarray]) -> float:
        """Identify the x-coordinate of the last topping."""
        # If there are no toppings or just one, return 0:
        if state.shape[1] <= 1:
            return 0
        x_coord = state[0, -1] / (self._viable_surface_radius * 2)
        return x_coord

    def y_coordinate(self, state: Union[list, np.ndarray]) -> float:
        """Identify the y-coordinate of the last topping."""
        # If there are no toppings or just one, return 0:
        if state.shape[1] <= 1:
            return 0
        y_coord = state[1, -1] / (self._viable_surface_radius * 2)
        return y_coord

    def distance_to_nearest_neighbor(
        self, state: Union[list, np.ndarray], normalize: bool = True
    ) -> float:
        """Compute the distance to the closest topping from the most recent."""
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
        if normalize:
            nearest = mags.min() / (self._viable_surface_radius * 2)
        else:
            nearest = mags.min()
        return nearest

    def dist_0_quadratic(self, state: Union[list, np.ndarray]) -> float:
        """Compute how close the distance between_toppings is to 0."""
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        dist = self.distance_to_nearest_neighbor(coords, normalize=False)
        quad = (dist - 0) ** 2
        return quad

    def dist_2_quadratic(self, state: Union[list, np.ndarray]) -> float:
        """Compute how close the distance between_toppings is to 2."""
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        dist = self.distance_to_nearest_neighbor(coords, normalize=False)
        quad = (dist - 2) ** 2
        return quad

    def dist_4_quadratic(self, state: Union[list, np.ndarray]) -> float:
        """Compute how close the distance between_toppings is to 4."""
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        dist = self.distance_to_nearest_neighbor(coords, normalize=False)
        quad = (dist - 4) ** 2
        return quad

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

    def make_pizza(self, learned_weights: np.ndarray) -> np.ndarray:
        """Make pizza according to learned_weights."""
        inner_reward = -np.inf

        toppings = generate_2D_points(self._viable_surface_radius, 1)
        new_toppings = generate_2D_points(
            self._viable_surface_radius, self._topping_sample_count
        )

        while toppings.shape[1] < self._max_topping_count:
            for i in range(new_toppings.shape[1]):
                temp_toppings = np.hstack(
                    (toppings, new_toppings[:, i].reshape(-1, 1))
                )
                features = (
                    self.compute_features(temp_toppings)
                    .squeeze()
                    .reshape(-1, 1)
                )
                new_reward = learned_weights @ features
                if new_reward > inner_reward:
                    # This placement of topping 'i' is better than all
                    # preceding placements; save but continue to look for a
                    # superior placement of 'i':
                    inner_reward = new_reward
                    inner_toppings = np.array(temp_toppings, copy=True)
                    topping_index = i
            toppings = np.array(inner_toppings, copy=True)
            inner_reward = -np.inf
            # Don't add the same topping more than once:
            new_toppings = np.delete(new_toppings, topping_index, axis=1)
        return toppings

    def visualize_pizza(self, toppings: np.ndarray) -> None:
        """Visualize a pizza."""
        fig, ax = plt.subplots()
        title = "Topping placements"
        ax.set_xlabel(title)
        ax.set_xlim(0, self._pizza_form["diameter"])
        ax.set_ylim(0, self._pizza_form["diameter"])
        dough = plt.Circle(
            (
                self._pizza_form["diameter"] / 2.0,
                self._pizza_form["diameter"] / 2.0,
            ),
            radius=self._pizza_form["diameter"] / 2.0,
            color="peru",
            fill=True,
        )

        cheese_radius = (
            self._pizza_form["diameter"] / 2.0
            - self._pizza_form["crust_thickness"]
        )
        cheese = plt.Circle(
            (
                self._pizza_form["diameter"] / 2.0,
                self._pizza_form["diameter"] / 2.0,
            ),
            radius=cheese_radius,
            color="lemonchiffon",
            fill=True,
        )
        latest_plot = ax.add_patch(dough)
        latest_plot = ax.add_patch(cheese)
        coords = toppings + self._pizza_form["diameter"] / 2.0

        # Plot the toppings. Currently using circle patches to easily
        # moderate the topping size:
        for i in range(coords.shape[1]):
            topping = plt.Circle(
                (coords[:, i]),
                radius=self._pizza_form["topping_diam"] / 2.0,
                color="firebrick",
                fill=True,
            )
            latest_plot = ax.add_patch(topping)
        for j in range(coords.shape[1]):
            ax.annotate(j, xy=(coords[0, j], coords[1, j]))
        latest_plot = ax.plot(coords[0, :], coords[1, :], "-b")

        plt.show()
        input("Press enter to continue.")

    def visualize_pizza_animated(
        self, toppings: np.ndarray, save_as_gif: bool = True
    ) -> None:
        """Visualize animated, stepwise pizza creation."""
        fig, ax = plt.subplots()
        diam = self._pizza_form["diameter"]
        t_size = float(self._pizza_form["topping_diam"])
        crust = self._pizza_form["crust_thickness"]
        coords = toppings + diam / 2.0
        ax.set_xlim(0, diam)
        ax.set_ylim(0, diam)

        # Create a base pizza and add to axes:
        dough = plt.Circle(
            (diam / 2.0, diam / 2.0),
            radius=diam / 2.0,
            color="peru",
            fill=True,
        )

        cheese_radius = diam / 2.0 - crust
        cheese = plt.Circle(
            (diam / 2.0, diam / 2.0),
            radius=cheese_radius,
            color="lemonchiffon",
            fill=True,
        )
        latest_plot = ax.add_patch(dough)
        latest_plot = ax.add_patch(cheese)

        # Plot the toppings. Currently using circle patches to easily
        # moderate the topping size:
        # for i in range(coords.shape[1]):
        def add_topping(coord):
            topping = plt.Circle(
                (coords[:, coord]),
                radius=t_size / 2.0,
                color="firebrick",
                fill=True,
            )
            latest_plot = ax.add_patch(topping)

        anim = FuncAnimation(
            fig, add_topping, frames=np.arange(coords.shape[1]), interval=200
        )
        if save_as_gif:
            anim_title = "topping_placements.gif"
            anim.save(anim_title, writer="pillow", fps=4)

        plt.show()
        input("Press enter to continue.")


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
