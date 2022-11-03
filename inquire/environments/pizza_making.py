"""A domain for real-world experiments."""
import time
from itertools import combinations
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from inquire.environments.environment import Environment
from inquire.utils.datatypes import Range, Trajectory
from matplotlib.animation import FuncAnimation
#from numba import jit


class PizzaMaking(Environment):
    """Create a pizza by placing toppings."""

    def __init__(
        self,
        seed: int = None,
        max_topping_count: int = 20,
        how_many_toppings_to_add: int = 1,
        topping_sample_count: int = 1000,
        pizza_form: dict = None,
        basis_functions: list = None,
        output_path: str = str(Path.cwd()) + "/output/pizza_making/",
        verbose: bool = False,
        save_png: bool = False,
        save_gif: bool = False,
    ):
        """Initialize a pizza-making domain.

        ::inputs:
            ::max_topping_count: Total number of toppings a pizza can have.
            ::how_many_toppings_to_add: Number of toppings added when
                                        generating pizzas.
            ::topping_sample_count: How many topping positions to sample during
                                    optimization.
            ::pizza_form: High-level attributes of a pizza.
            ::basis_functions: Functions which define features.
        """
        self._seed = seed
        self._which_reward = 3
        self._how_many_toppings_to_add = how_many_toppings_to_add
        self._max_topping_count = max_topping_count
        self._topping_sample_count = topping_sample_count
        self._pizza_form = pizza_form
        self._verbose = verbose
        self._save_png = save_png
        self._save_gif = save_gif
        self._output_path = Path(output_path)
        self.trajectory_length = 1
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)

        self._rng = np.random.default_rng(self._seed)
        crust_and_topping_factor = (self._pizza_form["crust_thickness"]) + (
            self._pizza_form["topping_diam"] / 2.0
        )
        self._viable_surface_radius = (
            self._pizza_form["diameter"] / 2.0 - crust_and_topping_factor
        )
        viable_surface_radius = self._viable_surface_radius
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

        action_low = np.full(
            shape=(2,), fill_value=-self._viable_surface_radius
        )
        action_high = np.full(
            shape=(2,), fill_value=self._viable_surface_radius
        )
        self._action_range = Range(
            action_low,
            np.ones_like(action_low),
            action_high,
            np.ones_like(action_high),
        )

        self._visualize_chosen_optima = False
        self._visualize_chosen_optima_animated = False
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
                basis_fn_memory_blocks.append(dist_0_quadratic)
            elif b == "dist_2_quadratic":
                basis_fn_memory_blocks.append(dist_2_quadratic)
            elif b == "dist_4_quadratic":
                basis_fn_memory_blocks.append(dist_4_quadratic)
            elif b == "distance_to_nearest_neighbor":
                basis_fn_memory_blocks.append(
                    self.distance_to_nearest_neighbor
                )
            elif b == "markovian_direction":
                basis_fn_memory_blocks.append(self.markovian_direction)
            elif b == "markovian_magnitude":
                basis_fn_memory_blocks.append(self.markovian_magnitude)
            elif b == "x_coordinate":
                basis_fn_memory_blocks.append(x_coordinate)
            elif b == "y_coordinate":
                basis_fn_memory_blocks.append(y_coordinate)

        # The feature function is a composition of the basis functions:

        def feature_fn(topping_coords: np.ndarray) -> np.ndarray:
            """Compute the features of a set of topping placements."""
            feature_values = np.empty((len(basis_fn_memory_blocks), ))
            # For each basis function in this feature function:
            if topping_coords.shape[1] <= 1:
                return np.zeros((len(basis_fn_memory_blocks), ))
            else:
                for i, b in enumerate(basis_fn_memory_blocks):
                    # Compute the feature value:
                    feature_val = b(viable_surface_radius, topping_coords)
                    feature_values[i] = feature_val
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
        if self._which_reward == 0:
            # Random reward:
            generated = self._rng.uniform(
                low=-1, high=1, size=(self._feature_count,)
            )
        elif self._which_reward == 1:
            # Favor left:
            generated = np.array([-10, 0, 0, 0])
        elif self._which_reward == 2:
            # Favor no overlap:
            generated = np.array([0, 0, 0, -80])
        elif self._which_reward == 3:
            # Favor left + no overlap:
            generated = np.array([-5, 0, 0, -80])
        generated = generated / np.linalg.norm(generated)
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
        best_toppings = np.array(start_state, copy=True)
        inner_toppings = None
        best_reward = -np.inf
        best_features = None
        # Generate a bunch of potential positions for toppings:
        position_candidates = generate_2D_points(
            self._viable_surface_radius, self._topping_sample_count * 50
        )
        for _ in range(self._how_many_toppings_to_add):
            # See which of position_candidates yields the greatest reward:
            for j in range(position_candidates.shape[1]):
                temp_toppings = np.hstack(
                    (best_toppings, position_candidates[:, j].reshape(-1, 1))
                )
                temp_features = self.compute_features(temp_toppings).squeeze()
                new_reward = w @ temp_features.T
                if new_reward > best_reward:
                    best_reward = new_reward
                    inner_toppings = np.array(temp_toppings, copy=True)
                    inner_features = np.array(temp_features, copy=True)
            best_reward = -np.inf
            best_toppings = np.array(inner_toppings, copy=True)
            best_features = inner_features

        best_topping_placements = Trajectory(
            states=best_toppings,
            actions=best_toppings[:, -1],
            phi=best_features,
        )
        if self._verbose:
            elapsed = time.perf_counter() - start
            print(
                f"It took {elapsed:.2f}s to find the best placement for the "
                "next topping."
            )
        if self._visualize_chosen_optima:
            optimal_placements = self.make_pizza(
                best_topping_placements.states
            )
            self.visualize_pizza(optimal_placements, self._save_png)
            if self._visualize_chosen_optima_animated:
                self.visualize_pizza_animated(
                    optimal_placements, self._save_gif
                )
        return best_topping_placements

    def action_space(self) -> Range:
        """Get the environment's action space."""
        return self._action_range

    def trajectory_rollout(
        self, start_state: np.ndarray, actions: np.ndarray
    ) -> Trajectory:
        """Create the trajectory from taking actions from start_state."""
        # A pizza's state is defined by its toppings' positions; since
        # actions == topping positions, the sequence of states is
        # start_state + actions:
        topping_positions = generate_2D_points(
            self._viable_surface_radius,
            count=start_state.reshape(2, -1).shape[1] + 1,
        )
        trajectory = Trajectory(
            states=np.concatenate(
                (
                    start_state.reshape(2, -1),
                    topping_positions[:, -1].reshape(2, 1),
                ),
                axis=1,
            ),
            actions=topping_positions[:, -1].reshape(2, 1),
            phi=None,
        )
        trajectory.phi = self.features_from_trajectory(trajectory)
        return trajectory

    def features_from_trajectory(self, trajectory: Trajectory) -> np.ndarray:
        """Compute the features for a series of topping placements.

        TODO: Consider features across trajectories > 1 step.
        """
        # We infer actions from the states; pass None in action's place:
        features = np.array(
            [
                self.compute_features(
                    trajectory.states[:, : (trajectory.states.shape[1] - i)]
                )
                for i in range(self._how_many_toppings_to_add)
            ]
        )
        if features.shape[0] > 1:
            features = features.sum(axis=0)
        return features.squeeze()

    def w_dim(self) -> np.ndarray:
        """Return the dimensionality environment's features."""
        return self._feature_count

    def state_space(self):
        """Observation space is continuous; return None."""
        return np.inf

    def trajectory_from_states(self, states, features):
        """Placeholder."""
        return Trajectory(states, np.sum(features, axis=0))

    def distance_between_trajectories(self, a, b):
        """Placeholder."""
        topping_a = a.states[:, -1]
        topping_b = b.states[:, -1]
        distance = np.sqrt(((topping_a - topping_b)**2).sum())
        return distance

    """

    The proceeding code consists of domain-specific helper functions. No part
    of the Inquire framework should need to be modified to accommodate these
    functions.

    """

    def approximate_surface_coverage(
        self, state: Union[list, np.ndarray]
    ) -> float:
        """Compute the approximate surface-area coverage.

        NOT in use.
        """
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

    def markovian_magnitude(self, state: Union[list, np.ndarray]) -> float:
        """Compute magnitude between latest topping and its precedecessor.

        NOT in use.
        """
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        x_dist = coords[0, -1] - coords[0, -2]
        y_dist = coords[1, -1] - coords[1, -2]
        mag = np.sqrt(x_dist ** 2 + y_dist ** 2)
        return mag

    def markovian_direction(self, state: Union[list, np.ndarray]) -> float:
        """Compute direction to the latest topping from its predecessor.

        Up ~ 90 degrees
        Down ~ -90 degrees
        Left ~ +/-180 degrees
        Right ~ 0 degrees

        NOT in use.
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

        return direction_in_degrees

    def x_coordinate(self, state: Union[list, np.ndarray]) -> float:
        """Identify the x-coordinate of the last topping."""
        # If there are no toppings, return 0:
        if state.shape[1] < 1:
            return 0
        x_coord = state[0, -1] / self._viable_surface_radius
        return x_coord

    def y_coordinate(self, state: Union[list, np.ndarray]) -> float:
        """Identify the y-coordinate of the last topping."""
        # If there are no toppings, return 0:
        if state.shape[1] < 1:
            return 0
        y_coord = state[1, -1] / self._viable_surface_radius
        return y_coord

    def distance_to_nearest_neighbor(
        self, state: Union[list, np.ndarray], normalize: bool = False
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
        max_diff = self._viable_surface_radius * 2
        quad = np.abs(dist - 0) / max_diff
        return quad

    def dist_2_quadratic(self, state: Union[list, np.ndarray]) -> float:
        """Compute how close the distance between_toppings is to 2.

        NOT in use.
        """
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        dist = self.distance_to_nearest_neighbor(coords, normalize=False)
        max_diff = self._viable_surface_radius * 2 - 2
        quad = np.abs(dist - 2) / max_diff
        return quad

    def dist_4_quadratic(self, state: Union[list, np.ndarray]) -> float:
        """Compute how close the distance between_toppings is to 4."""
        coords = np.array(state, copy=True)
        # If there are no toppings or just one, return 0:
        if coords.shape[1] <= 1:
            return 0
        dist = self.distance_to_nearest_neighbor(coords, normalize=False)
        max_diff = self._viable_surface_radius * 2 - 4
        quad = np.abs(dist - 4) / max_diff
        return quad

    def avg_magnitude_last_to_all(
        self, state: Union[list, np.ndarray]
    ) -> float:
        """Compute average magnitude between latest topping and all others.

        NOT in use.
        """
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
                    # preceding placements; save, but continue to look for a
                    # superior placement via the remaining sampled positions:
                    inner_reward = new_reward
                    inner_toppings = np.array(temp_toppings, copy=True)
                    topping_index = i
            toppings = np.array(inner_toppings, copy=True)
            inner_reward = -np.inf
            # Don't add the same topping more than once:
            new_toppings = np.delete(new_toppings, topping_index, axis=1)
        return toppings

    def visualize_trajectory(
        self, trajectory: Union[np.ndarray, Trajectory]
    ) -> None:
        """Alias visualize_pizza."""
        if isinstance(trajectory, Trajectory):
            states = trajectory.states
        else:
            states = trajectory
        self.visualize_pizza(states)

    def visualize_pizza(self, toppings: np.ndarray, annotate: bool = False) -> None:
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
        if annotate:
            for j in range(coords.shape[1]):
                ax.annotate(j, xy=(coords[0, j], coords[1, j]))
            latest_plot = ax.plot(coords[0, :], coords[1, :], "-b")

        if self._save_png:
            curr_time = time.strftime("%m:%d:%H:%M:%S", time.localtime())
            title = curr_time + "_pizza.png"
            plt.savefig(str(self._output_path) + "/" + title)
        plt.show()

    def visualize_pizza_animated(self, toppings: np.ndarray) -> None:
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
        plt.show()
        if self._save_gif:
            curr_time = time.strftime("%m:%d:%H:%M:%S", time.localtime())
            anim_title = curr_time + "_pizza.gif"
            anim.save(
                str(self._output_path) + "/" + anim_title,
                writer="pillow",
                fps=4,
            )


"""

The proceeding are Numba-optimized helper functions. No part of the Inquire
framework should need to be modified to accommodate these functions.

"""

'''
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


@jit(nopython=True)
def x_coordinate(radius: float, state: Union[list, np.ndarray]) -> float:
    """Identify the x-coordinate of the last topping."""
    x_coord = state[0, -1] / radius
    return x_coord


@jit(nopython=True)
def y_coordinate(radius: float, state: Union[list, np.ndarray]) -> float:
    """Identify the y-coordinate of the last topping."""
    y_coord = state[1, -1] / radius
    return y_coord


@jit(nopython=True)
def distance_to_nearest_neighbor(state: Union[list, np.ndarray]) -> float:
    """Compute the distance to the closest topping from the most recent."""
    most_recent = state[:, -1].copy()
    # Vectorize the last topping for efficient operations:
    most_recent = most_recent.reshape(2, 1)
    dists = np.empty((state.shape[1]-1, ))
    for i in range(dists.shape[0]):
        dist_a = most_recent[0, 0] - state[0, i]
        dist_b = most_recent[1, 0] - state[1, i]
        sq_dist_a = dist_a ** 2
        sq_dist_b = dist_b ** 2
        sum_dist = sq_dist_a + sq_dist_b
        dists[i] = sum_dist
    mags = np.sqrt(dists)
    nearest = mags.min()
    return nearest


@jit(nopython=True)
def dist_0_quadratic(radius: float, state: Union[list, np.ndarray]) -> float:
    """Compute how close the distance between_toppings is to 0."""
    coords = state
    dist = distance_to_nearest_neighbor(coords)
    max_diff = radius * 2
    quad = np.abs(dist - 0) / max_diff
    return quad


@jit(nopython=True)
def dist_2_quadratic(radius: float, state: Union[list, np.ndarray]) -> float:
    """Compute how close the distance between_toppings is to 2.

    NOT in use.
    """
    coords = state
    dist = distance_to_nearest_neighbor(coords)
    max_diff = radius * 2 - 2
    quad = np.abs(dist - 2) / max_diff
    return quad


@jit(nopython=True)
def dist_4_quadratic(radius: float, state: Union[list, np.ndarray]) -> float:
    """Compute how close the distance between_toppings is to 4."""
    coords = state
    dist = distance_to_nearest_neighbor(coords)
    max_diff = radius * 2 - 4
    quad = np.abs(dist - 4) / max_diff
    return quad
'''
