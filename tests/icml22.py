from inquire.environments import *
from inquire.agents import *
from inquire.teachers import *
from inquire.interactions.modalities import *
from inquire.utils.sampling import TrajectorySampling
from evaluation import Evaluation
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-V", "--verbose", dest='verbose', action='store_true',
                       help='verbose')
    parser.add_argument("-K", "--queries",  type=int, dest='num_queries', default=5,
                       help='number of queries')
    parser.add_argument("-R", "--runs", type=int, dest='num_runs', default=5,
                       help='number of evaluations to run')
    parser.add_argument("-X", "--tests", type=int, dest='num_test_states', default=10,
                       help='number of test states to evaluate')
    parser.add_argument("-Z", "--tasks", type=int, dest='num_tasks', default=1,
                       help='number of task instances to generate')
    parser.add_argument("-M", type=int, dest='num_w_samples', default=100,
                       help='number of weight samples')
    parser.add_argument("-N", type=int, dest='num_traj_samples', default=50,
                       help='number of trajectory samples')
    parser.add_argument("-D", "--domain", type=str, dest='domain_name', default="puddle", choices=["puddle"],
                       help='name of the evaluation domain')
    parser.add_argument("-S", "--sampling", type=str, dest='sampling_method', default="uniform", choices=["uniform", "uniform-without-duplicates", "rejection", "value_det", "value_prob"],
                       help='name of the trajectory sampling method')
    parser.add_argument("-A", "--agent", type=str, dest='agent_name', default="inquire", choices=["inquire", "demo-only", "pref-only", "corr-only"],
                       help='name of the agent to evaluate')
    parser.add_argument("-T", "--teacher", type=str, dest='teacher_name', default="optimal", choices=["optimal"],
                       help='name of the simulated teacher to query')

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
        sampling_params = ()
    elif args.sampling_method == "uniform-without-duplicates":
        sampling_method = TrajectorySampling.uniform_sampling
        sampling_params = tuple(["remove_duplicates=True"])

    ## Set up agent
    if args.agent_name == "inquire":
        agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration, Preference])
    elif args.agent_name == "demo-only":
        agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration])
    elif args.agent_name == "pref-only":
        agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Preference])
    elif args.agent_name == "corr-only":
        agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Correction])

    ## Set up teacher
    if args.teacher_name == "optimal":
        teacher = OptimalTeacher()

    ## Run evaluation ##
    Evaluation.run(domain, teacher, agent, args.num_tasks, args.num_runs, args.num_queries, args.num_test_states, args.verbose)
