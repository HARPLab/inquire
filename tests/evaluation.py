import pdb
import time

from inquire.environments.environment import Task

import numpy as np
from numpy.random import RandomState

class Evaluation:
    @staticmethod
    def run(domain, teacher, agent, num_tasks, num_runs, num_queries, num_test_states, verbose=False):
        rand = RandomState(0)
        perf_mat = np.zeros((num_tasks,num_runs,num_queries+1))
        dist_mat = np.zeros((num_tasks,num_runs,num_queries+1))
        if verbose:
            print("Initializing tasks...")
        tasks = [Task(domain, num_runs * num_queries, num_test_states, rand) for _ in range(num_tasks)]

        ## Each task is an instantiation of the domain (e.g., a particular reward function)
        for t in range(num_tasks):
            task_start = time.perf_counter()
            task = tasks[t]
            test_set = []
            state_idx = 0

            ## Establish baseline task performance using optimal trajectories
            if verbose:
                print("Finding optimal and worst-case baselines...")
            for test_state in task.test_states:
                best_traj = task.optimal_trajectory_from_ground_truth(test_state)
                max_reward = task.ground_truth_reward(best_traj)
                worst_traj = task.least_optimal_trajectory_from_ground_truth(test_state)
                min_reward = task.ground_truth_reward(worst_traj)
                test_set.append([test_state, (worst_traj, best_traj), (min_reward, max_reward)])

            ## Reset learning agent for each run and iterate through queries
            if verbose:
                print("Done. Starting queries...")
            run_start = time.perf_counter()
            agent.reset()
            for r in range(num_runs):
                if agent.__class__.__name__.lower() == "dempref":
                    demonstrations = []
                    for d in range(agent.n_demos):
                        random_start_state = task.query_states[
                            np.random.randint(low=0, high=len(task.query_states))
                        ]
                        demonstrations.append(
                            task.optimal_trajectory_from_ground_truth(
                              random_start_state
                            )
                        )
                    agent.process_demonstrations(demonstrations, domain)

                ## Record performance before first query
                perfs = []
                feedback = []
                w_dist = agent.initialize_weights(domain)
                w_mean = np.mean(w_dist, axis=0)
                for c in range(num_test_states):
                    model_traj = domain.optimal_trajectory_from_w(test_set[c][0], w_mean)
                    reward = task.ground_truth_reward(model_traj)
                    min_r, max_r = test_set[c][2]
                    perfs.append((reward - min_r) / (max_r - min_r))
                    # assert 0 <= perf <= 1
                perf_mat[t, r, 0] = np.mean(perfs)
                dist_mat[t, r, 0] = task.distance_from_ground_truth(w_mean)

                ## Iterate through queries
                for k in range(num_queries):
                    q_start = time.perf_counter()
                    if domain.__class__.__name__ == "LinearDynamicalSystem" or domain.__class__.__name__ == "LunarLander":
                        domain.reset(task.query_states[state_idx])
                    print("\nTask " + str(t+1) + "/" + str(num_tasks) + ", Run " + str(r+1) + "/" + str(num_runs) + ", Query " + str(k+1) + "/" + str(num_queries) + "     ", end='\n')

                    ## Generate query and learn from feedback
                    q = agent.generate_query(domain, task.query_states[state_idx], w_dist, verbose)
                    state_idx += 1
                    q.task = task
                    teacher_fb = teacher.query_response(q, verbose)
                    feedback.append(teacher_fb)
                    w_opt = np.mean(agent.update_weights(w_dist, domain, feedback), axis=0)
                    w_dist = agent.step_weights(w_dist, domain, feedback)
                    w_mean = np.mean(w_dist,axis=0)
                    ## Get performance metrics for each test-state after
                    ## each query and corresponding weight update:
                    perfs = []
                    for c in range(num_test_states):
                        model_traj = domain.optimal_trajectory_from_w(test_set[c][0], w_opt)
                        reward = task.ground_truth_reward(model_traj)
                        min_r, max_r = test_set[c][2]
                        perfs.append((reward - min_r) / (max_r - min_r))
                        # assert 0 <= perf <= 1
                    perf_mat[t, r, k+1] = np.mean(perfs)
                    dist_mat[t, r, k+1] = task.distance_from_ground_truth(w_opt)
                    q_time = time.perf_counter() - q_start
                    if verbose:
                        print(f"Query {k+1} in task {t+1}, run {r+1} took "
                              f"{q_time:.4}s to complete.")
                run_time = time.perf_counter() - run_start
                if verbose:
                    print(f"Run {r+1} in task {t+1} took {run_time:.4}s "
                           "to complete.")

            task_time = time.perf_counter() - task_start
            if verbose:
                print(f"Task {t+1} took {task_time:.4f}s to complete.")

        return perf_mat, dist_mat
