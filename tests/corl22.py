import argparse
import math
import os
import pdb
import time
import numpy as np

from inquire.environments import *
from inquire.agents import *
from inquire.teachers import *
from inquire.interactions.feedback import Modality
from inquire.utils.sampling import TrajectorySampling
from evaluation import Evaluation
from data_utils import save_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("-C", "--use_cache", dest='use_cache', action='store_true',
                       help='use cached trajectories instead of sampling')
    parser.add_argument("--teacher_displays", action="store_true",
                       help="display the teacher's interactions.")
    parser.add_argument("-K", "--queries",  type=int, dest='num_queries', default=5,
                       help='number of queries')
    parser.add_argument("-R", "--runs", type=int, dest='num_runs', default=10,
                       help='number of evaluations to run for each task')
    parser.add_argument("-X", "--tests", type=int, dest='num_test_states', default=1,
                       help='number of test states to evaluate')
    parser.add_argument("-Z", "--tasks", type=int, dest='num_tasks', default=1,
                       help='number of task instances to generate')
    parser.add_argument("-M", type=int, dest='num_w_samples', default=100,
                       help='number of weight samples')
    parser.add_argument("-N", type=int, dest='num_traj_samples', default=50,
                       help='number of trajectory samples')
    parser.add_argument("-D", "--domain", type=str, dest='domain_name', default="linear_combo", choices=["lander", "linear_combo", "linear_system", "gym_wrapper", "pizza"],
                       help='name of the evaluation domain')
    parser.add_argument("-I", "--opt_iterations", type=int, dest='opt_iters', default=50,
                       help='number of attempts to optimize a sample of controls (pertinent to lunar lander, linear system, and pizza-making domains)')
    parser.add_argument("-S", "--sampling", type=str, dest='sampling_method', default="uniform", choices=["uniform"],
                       help='name of the trajectory sampling method')
    parser.add_argument("-A", "--agent", type=str, dest='agent_name', default="inquire", choices=["inquire", "dempref", "demo-only", "pref-only", "corr-only", "binary-only", "all", "titrated", "inquire2"],
                       help='name of the agent to evaluate')
    parser.add_argument("-T", "--teacher", type=str, dest='teacher_name', default="optimal", choices=["optimal"],
                       help='name of the simulated teacher to query')
    parser.add_argument("-O", "--output", type=str, dest='output_dir', default="output",
                       help='name of the output directory')

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
    ## Set up sampling method
    if args.sampling_method == "uniform":
        sampling_method = TrajectorySampling.uniform_sampling
        sampling_params = {}
    else:
        raise ValueError("Unknown trajectory sampling method")

    ## Set up agent(s)
    if args.agent_name == "titrated":
        ddddd = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*5)
        ddddp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*4 + [Modality.PREFERENCE])
        dddpp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*3 + [Modality.PREFERENCE]*2)
        ddppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*2 + [Modality.PREFERENCE]*3)
        dpppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION] + [Modality.PREFERENCE]*4)
        ppppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.PREFERENCE]*5)
        agents = [ddddd, ddddp, dddpp, ddppp, dpppp, ppppp] 
        agent_names = ["DDDDD", "DDDDP", "DDDPP", "DDPPP", "DPPPP", "PPPPP"]
    if args.agent_name.lower() == "dempref":
        agents = [DemPref(
                weight_sample_count=args.num_w_samples,
                trajectory_sample_count=args.num_traj_samples,
                trajectory_length=traj_length,
                interaction_types=[Modality.DEMONSTRATION, Modality.PREFERENCE],
                w_dim=domain.w_dim,
                which_param_csv=0
                )]
        agent_names = ["DEMPREF"]
    if args.agent_name == "inquire":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION, Modality.PREFERENCE, Modality.CORRECTION, Modality.BINARY])]
        agent_names = ["INQUIRE"]
    elif args.agent_name == "demo-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION])]
        agent_names = ["Demo-only"]
    elif args.agent_name == "pref-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.PREFERENCE])]
        agent_names = ["Pref-only"]
    elif args.agent_name == "corr-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.CORRECTION])]
        agent_names = ["Corr-only"]
    elif args.agent_name == "binary-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.BINARY])]
        agent_names = ["Binary-only"]

    ## Set up teacher
    if args.teacher_name == "optimal":
        teacher = OptimalTeacher(
                     args.num_traj_samples, traj_length, args.teacher_displays
                  )

    ## Run evaluation ##
    all_perf, all_dist, all_query_types = [], [], []
    start = time.perf_counter()
    eval_time = time.strftime("/%m:%d:%H:%M", time.localtime())
    for agent, name in zip(agents, agent_names):
        print("Evaluating " + name + " agent...                    ")
        perf, dist, q_type = Evaluation.run(domain, teacher, agent, args.num_tasks, args.num_runs, args.num_queries, args.num_test_states, args.use_cache, args.verbose)
        all_perf.append(perf)
        all_dist.append(dist)
        all_query_types.append(q_type)
    elapsed = time.perf_counter() - start
    if args.verbose:
        print(f"The complete evaluation took {elapsed:.4} seconds.")
    eval_time = time.strftime("_%m:%d:%H:%M", time.localtime())
    save_data(
        all_dist,
        agent_names,
        args.num_runs,
        args.output_dir,
        domain.__class__.__name__ + eval_time + "_distance.csv"
    )
    save_data(
        all_perf,
        agent_names,
        args.num_runs,
        args.output_dir,
        domain.__class__.__name__ + eval_time + "_performance.csv"
    )
    save_data(
        all_query_types,
        agent_names,
        args.num_runs,
        args.output_dir,
        domain.__class__.__name__ + eval_time + "_queries.csv"
    )
