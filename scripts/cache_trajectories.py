import sys
import pickle
import pdb
import time
import os
import numpy as np
from inquire.utils.args_handler import ArgsHandler
from inquire.utils.sampling import TrajectorySampling, CachedSamples
from inquire.environments.environment import Task, CachedTask
from numpy.random import RandomState
from pathlib import Path

if __name__ == '__main__':
    rand = np.random.RandomState(0)
    args = ArgsHandler()
    domain = args.setup_domain()

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
    tasks = [Task(domain, 0, args.num_test_states, test_state_rand) for _ in range(args.num_tasks)]

    ## Each task is an instantiation of the domain (e.g., a particular reward function)
    for t in range(args.num_tasks):
        demos = []
        task = tasks[t]
        task_filename = prefix + str(t) + suffix
        print(task._r)
        print("Cacheing task " + str(t) + "/" + str(args.num_tasks))
        for s in range(args.num_test_states):
            print("\rState " + str(s) + "/" + str(args.num_test_states) + "     ")
            sys.stdout.flush()
            filename = prefix + str(t) + "_state-" + str(s) + suffix
            test_state = task.test_states[s]
            samples = TrajectorySampling.uniform_sampling(test_state, None, task.domain, rand, domain.trajectory_length, args.num_traj_samples, {})
            rewards = np.array([task.ground_truth_reward(s) for s in samples])
            demos.append(CachedSamples(task, test_state, samples[np.argmax(rewards)], samples[np.argmin(rewards)], samples))
            f_out = open(directory + filename, "wb")
            pickle.dump(demos[-1], f_out)
            f_out.close()
        print("")
    print("Done!")


