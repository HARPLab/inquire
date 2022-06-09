# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
from inquire.environments.pizza_making import PizzaMaking

form = {"diameter": 35, "crust_thickness": 2.54, "topping_diam": 3.54}
basis_fns = [
    "x_coordinate",  # Negative = prefer left
    "y_coordinate",  # Negative = prefer down
    "dist_0_quadratic",  # Negative = penalize distance-from-0
    "dist_4_quadratic",  # Negative = penalize distance-from-4
]
learned_weights = np.array([0, 0, 4.0, -4.0])
domain = PizzaMaking(pizza_form=form, basis_functions=basis_fns)
tops = domain.make_pizza(learned_weights)
domain.visualize_pizza(tops)


heavy_left = np.array([-20, 0, 0, 0])
heavy_overlap = np.array([0, 0, -80.0, 0])
heavy_left_no_overlap = np.array([-10, 0, 0, -80.0])
