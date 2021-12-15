import pdb
import numpy as np
from numpy.random import RandomState
from inquire.environments.environment import Task

class Evaluation:
    @staticmethod
    def run(domain, teacher, agent, num_tasks, num_runs, num_queries, num_test_states, verbose=False):
        rand = RandomState(0)
        perf_mat = np.zeros((num_tasks,num_runs,num_queries))
        dist_mat = np.zeros((num_tasks,num_runs,num_queries))
        if verbose:
            print("Initializing tasks...")
        tasks = [Task(domain, num_runs * num_queries, num_test_states, rand) for _ in range(num_tasks)]

        ## Each task is an instantiation of the domain (e.g., a particular reward function)
        for t in range(num_tasks):
            task = tasks[t]
            test_set = []
            state_idx = 0

            ## Establish baseline task performance using optimal trajectories
            if verbose:
                print("Finding optimal baselines...")
            for test_state in task.test_states:
                optimal_traj = task.optimal_trajectory_from_ground_truth(test_state)
                optimal_reward = task.ground_truth_reward(optimal_traj)
                test_set.append([test_state, optimal_traj, optimal_reward])

            ## Reset learning agent for each run and iterate through queries
            if verbose:
                print("Done. Starting queries...")
            for r in range(num_runs):
                agent.reset()
                w_dist = None
                feedback = []
                w_dist = agent.update_weights(domain, feedback)
                for k in range(num_queries):
                    print("Task " + str(t+1) + "/" + str(num_tasks) + ", Run " + str(r+1) + "/" + str(num_runs) + ", Query " + str(k+1) + "/" + str(num_queries) + "     ", end='\r') 
                    ## Generate query and learn from feedback
                    q = agent.generate_query(domain, task.query_states[state_idx], w_dist, verbose)
                    state_idx += 1
                    q.task = task
                    feedback.append(teacher.query(q))
                    w_dist = agent.update_weights(domain, feedback)
                    w_mean = np.mean(w_dist, axis=0)

                    ## Get performance metrics after weight update
                    perf = 0.0
                    for test_case in test_set:
                        model_traj = domain.optimal_trajectory_from_w(test_case[0], w_mean)
                        perf_mat[t,r,k] += task.ground_truth_reward(model_traj)/test_case[2]
                    perf_mat[t,r,k] = perf_mat[t,r,k]/float(len(test_set))
                    dist_mat[t,r,k] = task.distance_from_ground_truth(w_mean)
            pdb.set_trace()
        return perf_mat, dist_mat
