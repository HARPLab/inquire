import numpy as np
import argparse
from inquire.environments import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("-X", "--tests", type=int, dest='num_test_states', default=1,
                       help='number of test states to evaluate')
    parser.add_argument("-D", "--domain", type=str, dest='domain_name', default="linear_combo", choices=["lander", "linear_combo", "linear_system", "gym_wrapper", "pizza"], help='name of the evaluation domain')
    parser.add_argument("-I", "--opt_iterations", type=int, dest='opt_iters', default=50,
                       help='number of attempts to optimize a sample of controls (pertinent to lunar lander, linear system, and pizza-making domains)')
    parser.add_argument("-W", "--weights", type=str, dest='weights', default=None)
    args = parser.parse_args()

        ## Set up domain
    if args.domain_name == "linear_combo":
        traj_length = 1
        seed = 42
        w_dim = 32
        domain = LinearCombination(seed, w_dim)

    elif args.domain_name == "lander":
        traj_length = 10
        # Increase the opt_trajectory_iterations to improve optimization (but
        # increasing runtime as a consequence):
        optimization_iteration_count = args.opt_iters
        domain = LunarLander(
            optimal_trajectory_iterations=optimization_iteration_count,
            verbose=args.verbose
        )

    elif args.domain_name == "linear_system":
        traj_length = 15
        # Increase the opt_trajectory_iterations to improve optimization (but
        # increasing runtime as a consequence):
        optimization_iteration_count = args.opt_iters
        domain = LinearDynamicalSystem(
            trajectory_length=traj_length,
            optimal_trajectory_iterations=optimization_iteration_count,
            verbose=args.verbose
        )
    elif args.domain_name == "pizza":
        traj_length = 1
        max_topping_count = 30
        optimization_iteration_count = args.opt_iters
        pizza_form = {
            "diameter": 35,
            "crust_thickness": 2.54,
            "topping_diam": 3.54,
        }
        basis_functions = [
            "markovian_magnitude",
            "approximate_overlap_last_to_all",
            "avg_magnitude_last_to_all"
        ]
        domain = Pizza(
            max_topping_count=max_topping_count,
            optimization_iteration_count=500, #  optimization_iteration_count,
            pizza_form=pizza_form,
            basis_functions=basis_functions,
            verbose=args.verbose
        )

    rand = np.random.RandomState(0)
    weights = np.array([float(w) for w in args.weights.replace(',',' ').split()])
    if weights.shape[0] != domain.w_dim():
        raise AssertionError(domain.__class__.__name__ + " requires " + str(domain.w_dim()) + " weights")

    for _ in range(args.num_test_states):
        s = domain.generate_random_state(rand)
        traj = domain.optimal_trajectory_from_w(s, weights)
        domain.visualize_trajectory(s, traj)

