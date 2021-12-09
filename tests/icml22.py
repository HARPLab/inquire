from inquire.environments import *
from inquire.agents import *
from inquire.teachers import *
from evaluation import Evaluation
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for evaluating INQUIRE')
    parser.add_argument("-K", type=int, dest='num_queries', default=5,
                       help='number of queries')
    parser.add_argument("-R", type=int, dest='num_runs', default=5,
                       help='number of evaluations to run')
    parser.add_argument("-D", type=str, dest='domain_name', default="puddle", choices=["puddle"],
                       help='name of the evaluation domain')
    parser.add_argument("-A", type=str, dest='agent_name', default="inquire", choices=["inquire"],
                       help='name of the agent to evaluate')
    parser.add_argument("-T", type=str, dest='teacher_name', default="optimal", choices=["optimal"],
                       help='name of the simulated teacher to query')

    args = parser.parse_args()

    if args.domain_name == "puddle":
        domain = PuddleWorld()

    if args.agent_name == "inquire":
        agent = INQUIRE()

    if args.teacher_name == "optimal":
        teacher = OptimalTeacher()

    #Evaluation.run(domain, teacher, agent, args.num_runs, args.num_queries)
