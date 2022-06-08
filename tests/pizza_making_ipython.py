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
    "dist_2_quadratic",  # Negative = penalize distance-from-2
    "dist_4_quadratic",  # Negative = penalize distance-from-4
]
# Note that the features are NOT on the same scale
learned_weights = np.array([-1.99, -0.1, 4.0, 1.0, -4.0])
domain = PizzaMaking(pizza_form=form, basis_functions=basis_fns)
tops = domain.make_pizza(learned_weights)
domain.visualize_pizza(tops, save=True)
