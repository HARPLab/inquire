import pdb
import numpy as np
from numpy.random import RandomState
from inquire.environments.environment import Task

class Evaluation:
    @staticmethod
    def run(domain, teacher, agent, num_tasks, num_runs, num_queries, num_test_states, verbose=False):
        rand = RandomState(0)
        perf_mat = np.zeros((num_tasks,num_runs*num_test_states,num_queries))
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
                best_traj = task.optimal_trajectory_from_ground_truth(test_state)
                max_reward = task.ground_truth_reward(best_traj)
                worst_traj = task.least_optimal_trajectory_from_ground_truth(test_state)
                min_reward = task.ground_truth_reward(worst_traj)
                test_set.append([test_state, (worst_traj, best_traj), (min_reward, max_reward)])

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
                    teacher_fb = teacher.query(q, verbose)
                    if teacher_fb.selection is not None:
                        feedback.append(teacher_fb)
                    w_dist = agent.update_weights(domain, feedback)
                    w_mean = np.mean(w_dist, axis=0)

                    ## Get performance metrics after weight update
                    for c in range(num_test_states):
                        model_traj = domain.optimal_trajectory_from_w(test_set[c][0], w_mean)
                        reward = task.ground_truth_reward(model_traj)
                        min_r, max_r = test_set[c][2]
                        perf = (reward - min_r) / (max_r - min_r)
                        assert 0 <= perf <= 1
                        perf_mat[t,(r*num_runs)+c,k] = perf
                    dist_mat[t,r,k] = task.distance_from_ground_truth(w_mean)
        return perf_mat, dist_mat
