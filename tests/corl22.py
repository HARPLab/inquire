import argparse
import math
import os
import pdb
import time
import numpy as np

from inquire.utils.datatypes import Modality
from inquire.utils.sampling import TrajectorySampling
from evaluation import Evaluation
from data_utils import save_data, save_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("--use_cache", dest='use_cache', action='store_true',
                       help='use cached trajectories instead of sampling')
    parser.add_argument("--static_state", dest='static_state', action='store_true',
                       help='use the same state for all queries')
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
    parser.add_argument("--beta", type=float, dest='beta', default=1.0,
                       help='optimality parameter')
    parser.add_argument("--betas", type=str, dest='beta_vals',
                       help='optimality parameter, set for each interaction type')
    parser.add_argument("-C", "--convergence_threshold", type=float, dest='convergence_threshold', default=1.0e-1,
                       help='convergence threshold for estimating weight distribution')
    parser.add_argument("--alpha", type=float, dest='step_size', default=0.05,
                       help='step size for weight optimization')
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
    parser.add_argument("-A", "--agent", type=str, dest='agent_name', default="inquire", choices=["inquire", "dempref", "demo-only", "pref-only", "corr-only", "binary-only", "all", "titrated"],
                       help='name of the agent to evaluate')
    parser.add_argument("-T", "--teacher", type=str, dest='teacher_name', default="optimal", choices=["optimal"],
                       help='name of the simulated teacher to query')
    parser.add_argument("-O", "--output", type=str, dest='output_dir', default="output",
                       help='name of the output directory')
    parser.add_argument("--output_name", type=str, dest='output_name',
                       help='name of the output filename')
    parser.add_argument("-L", "--data_to_save", type=str, dest='data_to_save', default="distance,performance,query_types,dempref_metric",
                       help='list of which data to save for analysis')
    parser.add_argument("--seed_with_n_demos", type=int, dest="n_demos", default=1,
                       help="how many demos to provide before commencing preference queries. Specific to DemPref.")

    args = parser.parse_args()

    ## Set up domain
    if args.domain_name == "linear_combo":
        from inquire.environments.linear_combo import LinearCombination
        traj_length = 1
        seed = 42
        w_dim = 16
        domain = LinearCombination(seed, w_dim)

    elif args.domain_name == "lander":
        from inquire.environments.lunar_lander import LunarLander
        traj_length = 10
        optimization_iteration_count = args.opt_iters
        domain = LunarLander(
            optimal_trajectory_iterations=optimization_iteration_count,
            verbose=args.verbose
        )

    elif args.domain_name == "linear_system":
        from inquire.environments.linear_dynamical_system import LinearDynamicalSystem
        traj_length = 15
        optimization_iteration_count = args.opt_iters
        domain = LinearDynamicalSystem(
            trajectory_length=traj_length,
            optimal_trajectory_iterations=optimization_iteration_count,
            verbose=args.verbose
        )
    elif args.domain_name == "pizza":
        from inquire.environments.pizza_making import PizzaMaking
        traj_length = how_many_toppings_to_add = 1
        max_topping_count = 15
        pizza_form = {
            "diameter": 35,
            "crust_thickness": 2.54,
            "topping_diam": 3.54,
        }
        basis_functions = [
            "x_coordinate",
            "y_coordinate",
            "dist_0_quadratic",
            "dist_4_quadratic",
        ]
        domain = PizzaMaking(
            max_topping_count=max_topping_count,
            how_many_toppings_to_add=how_many_toppings_to_add,
            pizza_form=pizza_form,
            basis_functions=basis_functions,
            verbose=args.verbose,
        )
    ## Set up sampling method
    if args.sampling_method == "uniform":
        sampling_method = TrajectorySampling.uniform_sampling
        sampling_params = {}
    else:
        raise ValueError("Unknown trajectory sampling method")

    ## Set up agent(s)
    if args.agent_name == "titrated":
        from inquire.agents.inquire import FixedInteractions
        ddddd = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*5)
        ddddp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*4 + [Modality.PREFERENCE])
        dddpp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*3 + [Modality.PREFERENCE]*2)
        ddppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION]*2 + [Modality.PREFERENCE]*3)
        dpppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION] + [Modality.PREFERENCE]*4)
        ppppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.PREFERENCE]*5)
        agents = [ddddd, ddddp, dddpp, ddppp, dpppp, ppppp] 
        agent_names = ["DDDDD", "DDDDP", "DDDPP", "DDPPP", "DPPPP", "PPPPP"]
    if args.agent_name.lower() == "dempref":
        from inquire.agents.dempref import DemPref
        agents = [DemPref(
                weight_sample_count=args.num_w_samples,
                trajectory_sample_count=args.num_traj_samples,
                trajectory_length=traj_length,
                interaction_types=[Modality.DEMONSTRATION, Modality.PREFERENCE],
                w_dim=domain.w_dim(),
                seed_with_n_demos=args.n_demos
                )]
        agent_names = ["DEMPREF"]
    if args.beta_vals is None:
        beta = args.beta
    else:
        beta = eval(args.beta_vals)
    if args.agent_name == "inquire":
        from inquire.agents.inquire import Inquire
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION, Modality.PREFERENCE, Modality.CORRECTION, Modality.BINARY], beta)]
        agent_names = ["INQUIRE"]
    elif args.agent_name == "demo-only":
        from inquire.agents.inquire import Inquire
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.DEMONSTRATION], beta)]
        agent_names = ["Demo-only"]
    elif args.agent_name == "pref-only":
        from inquire.agents.inquire import Inquire
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.PREFERENCE], beta)]
        agent_names = ["Pref-only"]
    elif args.agent_name == "corr-only":
        from inquire.agents.inquire import Inquire
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.CORRECTION], beta)]
        agent_names = ["Corr-only"]
    elif args.agent_name == "binary-only":
        from inquire.agents.inquire import Inquire
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Modality.BINARY], beta)]
        agent_names = ["Binary-only"]

    ## Set up teacher
    if args.teacher_name == "optimal":
        from inquire.teachers.optimal import OptimalTeacher
        teacher = OptimalTeacher(
                     args.num_traj_samples, traj_length, args.teacher_displays
                  )

    ## Run evaluation ##
    data = {}
    data["distance"] = []
    data["performance"] = []
    data["query_types"] = []
    data["dempref_metric"] = []
    start = time.perf_counter()
    eval_start_time = time.strftime("/%m:%d:%H:%M", time.localtime())
    for agent, name in zip(agents, agent_names):
        print("Evaluating " + name + " agent...                    ")
        perf, dist, q_type, dempref_metric = Evaluation.run(domain, teacher, agent, args.num_tasks, args.num_runs, args.num_queries, args.num_test_states, args.step_size, args.convergence_threshold, args.use_cache, args.static_state, args.verbose)
        if args.output_name is not None:
            dist_sum = np.sum(dist)
            perf_sum = np.sum(perf)
            with open(args.output_dir + '/' + "overview.txt", "a+") as f:
                f.write(args.output_name + ", " + str(dist_sum) + ", " + str(perf_sum) + '\n')
        data["distance"].append(dist)
        data["performance"].append(perf)
        data["query_types"].append(q_type)
        data["dempref_metric"].append(dempref_metric)
    elapsed = time.perf_counter() - start
    if args.verbose:
        print(f"The complete evaluation took {elapsed:.4} seconds.")
    if args.output_name is None:
        name = domain.__class__.__name__ + eval_start_time
    else:
        name = args.output_name
    data_to_save = args.data_to_save.replace(" ", "").split(",")
    for d in data_to_save:
        save_data(
            data=data[d],
            labels=agent_names,
            num_runs=args.num_runs,
            directory=args.output_dir,
            filename=name + f"_{d}.csv",
            subdirectory=domain.__class__.__name__
        )
    try:
        save_plot(
            data["distance"],
            agent_names,
            "w distance",
            [0,1],
            args.output_dir,
            name + "_distance.png",
            subdirectory=domain.__class__.__name__
        )
    except:
        print("save_plot() didn't work.")
        exit()
