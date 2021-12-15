import numpy as np
from numpy.random import RandomState
from inquire.environments.environment import Task

class Evaluation:
    @staticmethod
    def run(domain, teacher, agent, num_tasks, num_runs, num_queries, num_test_states):
        rand = RandomState(0)
        perf_mat = np.zeros((num_tasks,num_runs,num_queries))
        dist_mat = np.zeros((num_tasks,num_runs,num_queries))
        tasks = [Task(domain, num_runs * num_queries, num_test_states, rand) for _ in range(num_tasks)]

        ## Each task is an instantiation of the domain (e.g., a particular reward function)
        for t in range(num_tasks):
            task = tasks[t]
            state_idx = 0

            ## Establish baseline task performance using optimal trajectories
            for test_state in task.test_states:
                optimal_traj = task.optimal_trajectory_from_ground_truth(test_state)
                test_set.append([test_state, optimal_traj])

            ## Reset learning agent for each run and iterate through queries
            for r in range(num_runs):
                curr_agent = agent.spawn_agent()
                feedback = []
                for k in range(num_queries):
                    ## Generate query and learn from feedback
                    q = curr_agent.generate_query(domain, task.query_states[state_idx])
                    feedback.append(teacher.query(q))
                    w_dist = curr_agent.update_weights(domain, feedback)
                    w_mean = np.mean(w_dist, axis=1)

                    ## Get performance metrics after weight update
                    perf = 0.0
                    for test_case in test_set:
                        model_traj = domain.optimal_trajectory_from_w(test_case[0], w_mean)
                        perf_mat[t,r,k] += model_traj.phi/test_case[1].phi
                    perf_mat[t,r,k] = perf_mat[t,r,k]/float(len(test_set))
                    dist_mat[t,r,k] = task.distance_from_ground_truth(w_mean)
        return perf_mat, dist_mat
