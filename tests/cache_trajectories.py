import sys
import argparse
import pickle
import pdb
import time
import os
import numpy as np
import inquire.utils.sampling
from inquire.utils.sampling import TrajectorySampling, CachedSamples
from inquire.environments import *
from numpy.random import RandomState
from pathlib import Path

class CacheTrajectories:
    @staticmethod
    def run(domain, num_tasks, num_test_states, rand, steps, N, verbose=False):
        ## Setup filename and directory
        directory = "cache/"
        prefix = domain.__class__.__name__ + "_task-" 
        suffix = "_cache.pkl"
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)

        ## Instantiate tasks and states
        print("Initializing tasks...")
        test_state_rand = RandomState(0)
        tasks = [Task(domain, 0, num_test_states, test_state_rand) for _ in range(num_tasks)]

        ## Each task is an instantiation of the domain (e.g., a particular reward function)
        for t in range(num_tasks):
            demos = []
            task = tasks[t]
            task_filename = prefix + str(t) + suffix
            print(task._r)
            print("Cacheing task " + str(t) + "/" + str(num_tasks))
            for s in range(num_test_states):
                print("\rState " + str(s) + "/" + str(num_test_states) + "     ")
                sys.stdout.flush()
                filename = prefix + str(t) + "_state-" + str(s) + suffix
                test_state = task.test_states[s]
                samples = TrajectorySampling.uniform_sampling(test_state, None, task.domain, rand, steps, N, {})
                rewards = np.array([task.ground_truth_reward(s) for s in samples])
                demos.append(CachedSamples(task, test_state, samples[np.argmax(rewards)], samples[np.argmin(rewards)], samples))
                f_out = open(directory + filename, "wb")
                pickle.dump(demos[-1], f_out)
                f_out.close()
            print("")
            #task_file = open(directory + task_filename, "wb")
            #pickle.dump(demos, task_file)
            #task_file.close()
        print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("-X", "--tests", type=int, dest='num_test_states', default=1,
                       help='number of test states to evaluate')
    parser.add_argument("-Z", "--tasks", type=int, dest='num_tasks', default=1,
                       help='number of task instances to generate')
    parser.add_argument("-D", "--domain", type=str, dest='domain_name', default="puddle", choices=["puddle", "lander", "linear_combo", "linear_system", "gym_wrapper", "pizza"],
                       help='name of the evaluation domain')
    parser.add_argument("-I", "--opt_iterations", type=int, dest='opt_iters', default=50,
                       help='number of attempts to optimize a sample of controls (pertinent to lunar lander, linear system, and pizza-making domains)')
    parser.add_argument("-N", type=int, dest='num_traj_samples', default=50,
                       help='number of trajectory samples')

    args = parser.parse_args()

    ## Set up domain
    if args.domain_name == "puddle":
        traj_length = 20
        grid_dim = 8
        num_puddles = 10
        domain = PuddleWorld(grid_dim, traj_length, num_puddles)

    elif args.domain_name == "linear_combo":
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
    CacheTrajectories.run(domain, args.num_tasks, args.num_test_states, rand, traj_length, args.num_traj_samples, args.verbose)

