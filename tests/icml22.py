from inquire.environments import *
from inquire.agents import *
from inquire.teachers import *
from inquire.interactions.modalities import *
from inquire.utils.sampling import TrajectorySampling
from evaluation import Evaluation
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import argparse
import math

def plot_results(results, labels, dir_name, filename):
    colors = ['r','b','g','c','m','y','k']
    task_mat = np.stack(results, axis=1)
    file_path = os.path.realpath(__file__)
    output_dir = os.path.dirname(file_path) + "/" + dir_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(task_mat.shape[0]):
        for a in range(task_mat.shape[1]):
            series = np.transpose(task_mat[t,a])
            label = labels[a]
            x = [i+1+(0.05*a) for i in range(series.shape[0])]
            med = np.median(series,axis=1)
            err = abs(np.percentile(series,(25,75),axis=1)-med)
            plt.errorbar(x, med, fmt='.-', yerr=err, color=colors[a%len(colors)],label=label)
        plt.legend(labels)
        plt.xticks(np.arange(1, task_mat.shape[-1]+1, 1.0))
        plt.savefig(output_dir + filename + "-task_" + str(t) + ".png")
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("-K", "--queries",  type=int, dest='num_queries', default=5,
                       help='number of queries')
    parser.add_argument("-R", "--runs", type=int, dest='num_runs', default=10,
                       help='number of evaluations to run')
    parser.add_argument("-X", "--tests", type=int, dest='num_test_states', default=20,
                       help='number of test states to evaluate')
    parser.add_argument("-Z", "--tasks", type=int, dest='num_tasks', default=1,
                       help='number of task instances to generate')
    parser.add_argument("-M", type=int, dest='num_w_samples', default=100,
                       help='number of weight samples')
    parser.add_argument("-N", type=int, dest='num_traj_samples', default=50,
                       help='number of trajectory samples')
    parser.add_argument("-D", "--domain", type=str, dest='domain_name', default="puddle", choices=["puddle"],
                       help='name of the evaluation domain')
    parser.add_argument("-S", "--sampling", type=str, dest='sampling_method', default="uniform",
                       help='name of the trajectory sampling method')
    parser.add_argument("-A", "--agent", type=str, dest='agent_name', default="inquire", choices=["inquire", "demo-only", "pref-only", "corr-only", "all", "titrated"],
                       help='name of the agent to evaluate')
    parser.add_argument("-T", "--teacher", type=str, dest='teacher_name', default="optimal", choices=["optimal"],
                       help='name of the simulated teacher to query')
    parser.add_argument("-O", "--output", type=str, dest='output_dir', default="output",
                       help='name of the output directory')

    args = parser.parse_args()

    ## Set up domain
    if args.domain_name == "puddle":
        traj_length = 20
        grid_dim = 8
        num_puddles = 10
        domain = PuddleWorld(grid_dim, traj_length, num_puddles)

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
    else:
        raise ValueError("Unknown trajectory sampling method")

    ## Set up agent(s)
    if args.agent_name == "all":
        inquire_agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration, Preference])
        demo_agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration])
        pref_agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Preference])
        corr_agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Correction])
        agents = [demo_agent, pref_agent] #, inquire_agent, corr_agent]
        agent_names = ["Demo-only", "Pref-only"] #, "INQUIRE", "Corr-only"]
    if args.agent_name == "titrated":
        ddddd = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration]*5)
        ddddp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration]*4 + [Preference])
        dddpp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration]*3 + [Preference]*2)
        ddppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration]*2 + [Preference]*3)
        dpppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration] + [Preference]*4)
        ppppp = FixedInteractions(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Preference]*5)
        agents = [ddddd, ddddp, dddpp, ddppp, dpppp, ppppp] 
        agent_names = ["DDDDD", "DDDDP", "DDDPP", "DDPPP", "DPPPP", "PPPPP"]
    if args.agent_name == "inquire":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration, Preference])]
        agent_names = ["INQUIRE"]
    elif args.agent_name == "demo-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration])]
        agent_names = ["Demo-only"]
    elif args.agent_name == "pref-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Preference])]
        agent_names = ["Pref-only"]
    elif args.agent_name == "corr-only":
        agents = [Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Correction])]
        agent_names = ["Corr-only"]

    ## Set up teacher
    if args.teacher_name == "optimal":
        teacher = OptimalTeacher()

    ## Run evaluation ##
    all_perf, all_dist = [], []
    for agent, name in zip(agents, agent_names):
        print("Evaluating " + name + " agent...                    ")
        perf, dist = Evaluation.run(domain, teacher, agent, args.num_tasks, args.num_runs, args.num_queries, args.num_test_states, args.verbose)
        all_perf.append(perf)
        all_dist.append(dist)
    plot_results(all_perf, agent_names, args.output_dir, "performance")
    plot_results(all_dist, agent_names, args.output_dir, "distance")
