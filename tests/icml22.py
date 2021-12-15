from inquire.environments import *
from inquire.agents import *
from inquire.teachers import *
from inquire.interactions.modalities import *
from inquire.utils.sampling import TrajectorySampling
from evaluation import Evaluation
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-K", type=int, dest='num_queries', default=5,
                       help='number of queries')
    parser.add_argument("-R", type=int, dest='num_runs', default=5,
                       help='number of evaluations to run')
    parser.add_argument("-X", type=int, dest='num_test_states', default=10,
                       help='number of test states to evaluate')
    parser.add_argument("-Z", type=int, dest='num_tasks', default=1,
                       help='number of task instances to generate')
    parser.add_argument("-M", type=int, dest='num_w_samples', default=100,
                       help='number of weight samples')
    parser.add_argument("-N", type=int, dest='num_traj_samples', default=50,
                       help='number of trajectory samples')
    parser.add_argument("-D", type=str, dest='domain_name', default="puddle", choices=["puddle"],
                       help='name of the evaluation domain')
    parser.add_argument("-S", type=str, dest='sampling_method', default="uniform", choices=["uniform", "rejection", "value_det", "value_prob"],
                       help='name of the trajectory sampling method')
    parser.add_argument("-A", type=str, dest='agent_name', default="inquire", choices=["inquire"],
                       help='name of the agent to evaluate')
    parser.add_argument("-T", type=str, dest='teacher_name', default="optimal", choices=["optimal"],
                       help='name of the simulated teacher to query')

    args = parser.parse_args()

    if args.domain_name == "puddle":
        traj_length = 20
        grid_dim = 8
        num_puddles = 10
        domain = PuddleWorld(grid_dim, traj_length, num_puddles)

    if args.sampling_method == "uniform":
        sampling_method = TrajectorySampling.uniform_sampling
        sampling_params = ()

    if args.agent_name == "inquire":
        agent = Inquire(sampling_method, sampling_params, args.num_w_samples, args.num_traj_samples, traj_length, [Demonstration, Preference, Correction])

    if args.teacher_name == "optimal":
        teacher = OptimalTeacher()

    Evaluation.run(domain, teacher, agent, args.num_tasks, args.num_runs, args.num_queries, args.num_test_states)
