from inquire.environments import *
from inquire.utils.sampling import TrajectorySampling
from numpy.random import RandomState
import numpy as np
import os
import pdb
import argparse
import math
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("-N", type=int, dest='num_traj_samples', default=50,
                       help='number of trajectory samples')
    parser.add_argument("-D", "--domain", type=str, dest='domain_name', default="puddle", choices=["puddle", "lander", "gym_wrapper"],
                       help='name of the evaluation domain')
    parser.add_argument("-S", "--sampling", type=str, dest='sampling_method', default="uniform",
                       help='name of the trajectory sampling method')
    parser.add_argument("-O", "--output", type=str, dest='output_dir', default="output",
                       help='name of the output directory')

    args = parser.parse_args()

    ## Set up domain
    if args.domain_name == "puddle":
        traj_length = 20
        grid_dim = 8
        num_puddles = 10
        domain = PuddleWorld(grid_dim, traj_length, num_puddles)

    if args.domain_name == "lander":
        traj_length = 10
        # Increase the opt_trajectory_iterations to improve optimization:
        opt_trajectory_iterations = args.opt_iters
        domain = LunarLander(
            optimal_trajectory_iterations=opt_trajectory_iterations
        )

    ## Set up sampling method
    if args.sampling_method == "uniform":
        sampling_method = TrajectorySampling.uniform_sampling
        sampling_params = {}
    elif args.sampling_method == "uniform-without-duplicates":
        sampling_method = TrajectorySampling.uniform_sampling
        sampling_params = {"remove_duplicates":True, "timeout":30}
    elif args.sampling_method == "value-det-without-duplicates":
        sampling_method = TrajectorySampling.value_sampling
        sampling_params = {"remove_duplicates":True, "probabilistic":False, "timeout":30}
    elif args.sampling_method == "value-det":
        sampling_method = TrajectorySampling.value_sampling
        sampling_params = {"remove_duplicates":False, "probabilistic":False}
    elif args.sampling_method == "value-prob-without-duplicates":
        sampling_method = TrajectorySampling.value_sampling
        sampling_params = {"remove_duplicates":True, "probabilistic":True, "timeout":30}
    elif args.sampling_method == "value-prob":
        sampling_method = TrajectorySampling.value_sampling
        sampling_params = {"remove_duplicates":False, "probabilistic":True}
    elif args.sampling_method == "rejection-without-duplicates":
        sample_size = args.num_traj_samples * 5
        sampling_method = TrajectorySampling.percentile_rejection_sampling
        sampling_params = {"remove_duplicates":True, "sample_size":sample_size, "timeout":30}
    elif args.sampling_method == "rejection":
        sample_size = args.num_traj_samples * 5
        sampling_method = TrajectorySampling.percentile_rejection_sampling
        sampling_params = {"remove_duplicates":False, "sample_size":sample_size}
    elif args.sampling_method == "mcmc":
        sampling_method = TrajectorySampling.mcmc_sampling
        sampling_params = {}
    else:
        raise ValueError("Unknown trajectory sampling method")

    rand = RandomState(0)
    task = Task(domain, 1, 1, rand) 
    r = np.array([-0.5, 1.0, -4.0])
    w = r / np.linalg.norm(r)

    for test_state in task.test_states:
        samples = TrajectorySampling.mcmc_sampling(test_state, [w], domain, rand, domain.max_duration, 20, sampling_params)
