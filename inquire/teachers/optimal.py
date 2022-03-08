from inquire.teachers.teacher import Teacher
from inquire.interactions.feedback import Trajectory, Choice, Query
from inquire.interactions.modalities import Demonstration, Preference, Correction, BinaryFeedback
from inquire.utils.viz import Viz
from inquire.utils.sampling import TrajectorySampling
import inquire.utils.learning
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time # TODO remove

class OptimalTeacher(Teacher):
    @property
    def alpha(self):
        return self._alpha

    def __init__(self, N, steps) -> None:
        super().__init__()
        self._alpha = 0.95
        self._N = N
        self._steps = steps

    def query(self, q: Query, verbose: bool=False) -> Choice:
        # f = self.binary_feedback(q)
        # print(f.selection, f.options)
        # viz = Viz(q.trajectories[0].trajectory)
        # while not viz.exit:
        #     viz.draw()
        # assert(False)
        if q.query_type is Demonstration:
            f = self.demonstration(q)
            if verbose:
                print("Showing demonstrated trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
            return f
        elif q.query_type is Preference:
            f = self.preference(q)
            if verbose:
                print("Showing preferred trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
                for t in f.options:
                    if t is not f.selection:
                        print("Showing UNpreferred trajectory")
                        Viz.display_trajectory(t)
                        viz = Viz(t.trajectory)
                        while not viz.exit:
                            viz.draw()
            return f
        elif q.query_type is Correction:
            f = self.correction(q)
            if verbose:
                print("Showing corrected trajectory")
                Viz.display_trajectory(f.selection)
                for t in f.options:
                    if t is not f.selection:
                        print("Showing original trajectory")
                        Viz.display_trajectory(t)
                return f
        elif q.query_type is BinaryFeedback:
            f = self.binary_feedback(q, verbose)
            if verbose:
                print("Teacher Feedback: {}; Options: {}".format(f.selection, f.options))
                print("Showing original trajectory")
                viz = Viz(q.trajectories[0].trajectory)
                while not viz.exit:
                    viz.draw()
        else:
            raise Exception(self._type.__name__ + " does not support queries of type " + str(q.query_type))

    def demonstration(self, query: Query) -> Choice:
        traj = query.task.domain.optimal_trajectory_from_w(query.start_state, query.task.get_ground_truth())
        return Choice(traj, [traj] + query.trajectories)

    def preference(self, query: Query) -> Choice:
        r = [np.dot(query.task.get_ground_truth(), qi.phi) for qi in query.trajectories]
        return Choice(query.trajectories[np.argmax(r)], query.trajectories)

    def correction(self, query: Query) -> Choice:
        curr_state = query.start_state
        feats = [query.task.domain.features(None,curr_state)]
        values = inquire.utils.learning.value_iteration(query.task, query.task.get_ground_truth(), query.start_state)
        traj = [[None,query.start_state]]
        comp_traj = query.trajectories[0]
        for step in range(len(comp_traj)-1):
            opt_action = domain.available_actions[np.argmax(values[curr_state[0][0],curr_state[0][1]])]
            comp_state = comp_traj.traj[step+1]
            next_state = domain.next_state(curr_state, opt_action)
            next_action = opt_action
            if comp_traj.traj[step][1] == curr_state:
                # Still following proposed trajectory
                comp_x, comp_y = comp_state[1][0]
                comp_val = np.max(values[comp_x,comp_y])
                ## Change to sample using Sampling.weighted_choice
                #opt_prob = (np.max(action_vals) - comp_val) / np.std(action_vals)

                max_prob = scipy.stats.norm.cdf(np.max(action_vals), loc=np.mean(action_vals), scale=np.std(action_vals))
                comp_prob = scipy.stats.norm.cdf(comp_val, loc=np.mean(action_vals), scale=np.std(action_vals))
                #if len(accepted_r) == 0 or np.random.rand() < np.mean(r)/(M*np.mean(accepted_r)):
                #if True: #len(accepted_r) == 0 or np.random.rand() > (np.mean(accepted_r) - np.mean(r))/np.mean(accepted_r):
                #if np.random.normal(0,1) >= opt_prob:
                if np.random.rand() >= max_prob - comp_prob:
                    next_state = comp_state[1]
                    next_action = comp_state[0]
            curr_state = next_state
            action = next_action

            traj.append([action,curr_state])
            feats.append(domain.features(action,curr_state))
        resp = Trajectory(traj, np.sum(feats,axis=0))
        return Choice(resp, [resp] + query.trajectories)

    def binary_feedback(self, query: Query, verbose: bool=False) -> Choice:
        assert(len(query.trajectories) == 1)

        print('Probabilistic')
        start_time = time.time()
        traj_samples = TrajectorySampling.value_sampling(query.start_state, [query.task.get_ground_truth()], query.task.domain, np.random.RandomState(0), self._steps, self._N, {'remove_duplicates': True, 'probabilistic': True})
        print('Sampling took: {} seconds'.format(time.time()-start_time))
        rewards = np.array([np.dot(t.phi, query.task.get_ground_truth()) for t in traj_samples])
        # Construct CDF
        rewards = np.sort(rewards)
        rewards_cdf = np.linspace(0, 1, self._N)

        # Compare with ground truth optimal trajectory (TODO delete)
        best_traj = query.task.domain.optimal_trajectory_from_w(query.start_state, query.task.get_ground_truth())
        best_reward = np.dot(best_traj.phi, query.task.get_ground_truth())
        print('Best reward: {}'.format(best_reward))

        percentile_idx = np.argwhere(rewards_cdf >= self._alpha)[0,0]
        threshold_reward = rewards[percentile_idx]
        query_reward = np.dot(query.task.get_ground_truth(), query.trajectories[0].phi)
        print('Query reward: {}, Threshold reward: {}'.format(query_reward, threshold_reward))

        # Plot CDF
        if verbose:
            print(rewards)
            print(rewards_cdf)
            plt.figure()
            plt.plot(rewards, rewards_cdf)
            plt.title('Rewards CDF for Probabilistic Value Sampling')

        # Sample trajectories from uniform sampling
        print('Uniform')
        start_time = time.time()
        traj_samples = TrajectorySampling.uniform_sampling(query.start_state, None, query.task.domain, np.random.RandomState(0), self._steps, self._N, {'remove_duplicates': True})
        print('Sampling took: {} seconds'.format(time.time()-start_time))
        rewards = np.array([np.dot(t.phi, query.task.get_ground_truth()) for t in traj_samples])

        # Construct CDF
        rewards = np.sort(rewards)
        rewards_cdf = np.linspace(0, 1, self._N)

        # Compare with ground truth optimal trajectory (TODO delete)
        best_traj = query.task.domain.optimal_trajectory_from_w(query.start_state, query.task.get_ground_truth())
        best_reward = np.dot(best_traj.phi, query.task.get_ground_truth())
        print('Best reward: {}'.format(best_reward))

        percentile_idx = np.argwhere(rewards_cdf >= self._alpha)[0,0]
        threshold_reward = rewards[percentile_idx]
        query_reward = np.dot(query.task.get_ground_truth(), query.trajectories[0].phi)
        print('Query reward: {}, Threshold reward: {}'.format(query_reward, threshold_reward))

        # Plot CDF
        if verbose:
            print(rewards)
            print(rewards_cdf)
            plt.figure()
            plt.plot(rewards, rewards_cdf)
            plt.title('Rewards CDF for Uniform Sampling')
            plt.show()

        return Choice(1 if query_reward > threshold_reward else -1, [-1, 1])
