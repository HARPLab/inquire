import argparse
import math
import os
import pdb
import time
import numpy as np
from args_handler import ArgsHandler
from evaluation import Evaluation
from data_utils import save_data, save_plot

if __name__ == "__main__":
    args = ArgsHandler()
    domain = args.setup_domain()
    teacher = args.setup_teacher()
    agents, agent_names = args.setup_agents()

    ## Run evaluation ##
    data = {}
    data["distance"] = []
    data["performance"] = []
    data["query_types"] = []
    data["dempref_metric"] = []
    start = time.perf_counter()
    eval_start_time = time.strftime("_%m:%d:%H:%M", time.localtime())
    for agent, name in zip(agents, agent_names):
        print("Evaluating " + name + " agent...                    ")
        perf, dist, q_type, dempref_metric = Evaluation.run(domain, teacher, agent, args.num_tasks, args.num_runs, args.num_queries, args.num_test_states, args.step_size, args.conv_threshold, args.use_cache, args.static_state, args.verbose)
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
    for d in list(data.keys()):
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
