from inquire.teachers.teacher import Teacher
from inquire.interactions.feedback import Trajectory, Choice, Query
from inquire.interactions.modalities import Demonstration, Preference, Correction, BinaryFeedback
from inquire.utils.viz import Viz
from inquire.utils.sampling import TrajectorySampling
import inquire.utils.learning
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pdb

class OptimalTeacher(Teacher):
    @property
    def alpha(self):
        return self._alpha

    def __init__(self, N, steps, display_interactions: bool = False) -> None:
        super().__init__()
        self._alpha = 0.95
        self._N = N
        self._steps = steps
        self._display_interactions = display_interactions

    def query_response(self, q: Query, verbose: bool=False) -> Choice:
        if q.query_type is Demonstration:
            f = self.demonstration(q)
            if self._display_interactions:
                print("Showing demonstrated trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
            return f
        elif q.query_type is Preference:
            f = self.preference(q)
            if self._display_interactions:
                print("Showing preferred trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
                for t in f.options:
                    if t is not f.selection:
                        print("Showing UNpreferred trajectory")
                        Viz.visualize_trajectory(t)
                        viz = Viz(t.trajectory)
                        while not viz.exit:
                            viz.draw()
            return f
        elif q.query_type is Correction:
            f = self.correction(q)
            if self._display_interactions:
                print("Showing corrected trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
                for t in f.options:
                    if t is not f.selection:
                        print("Showing original trajectory")
                        viz = Viz(t.trajectory)
                        while not viz.exit:
                            viz.draw()
            return f
        elif q.query_type is BinaryFeedback:
            f = self.binary_feedback(q, verbose)
            if verbose:
                print("Teacher Feedback: {}".format(1 if f.selection is not None else -1))
            if self._display_interactions:
                print("Teacher Feedback: {}".format(1 if f.selection is not None else -1))
                print("Showing original trajectory")
                viz = Viz(q.trajectories[0].trajectory)
                while not viz.exit:
                    viz.draw()
            return f
        else:
            raise Exception(self._type.__name__ + " does not support queries of type " + str(q.query_type))

    def demonstration(self, query: Query) -> Choice:
        traj = query.task.domain.optimal_trajectory_from_w(query.start_state, query.task.get_ground_truth())
        return Choice(traj, [traj] + query.trajectories)

    def preference(self, query: Query) -> Choice:
        r = [query.task.ground_truth_reward(qi) for qi in query.trajectories]
        return Choice(selection=query.trajectories[np.argmax(r)], options=query.trajectories)

    def correction(self, query: Query) -> Choice:
        alpha = 1.0 #optional parameter for discouraging more distanced corrections
        rewards, dists = [], []
        traj_samples = TrajectorySampling.uniform_sampling(query.start_state, None, query.task.domain, np.random.RandomState(0), self._steps, self._N, {'remove_duplicates': True})
        t_query = query.trajectories[0]
        for t in traj_samples:
            t_reward = query.task.ground_truth_reward(t)
            rewards.append(t_reward)
            t_dist = query.task.domain.distance_between_trajectories(t, t_query)
            dists.append(t_dist)
        min_r = query.task.ground_truth_reward(t_query)
        max_r = np.max(np.array(rewards))
        scaled_rewards = np.array([(r - min_r) / (max_r - min_r) for r in rewards])
        dists = np.array(dists)
        scaled_dists = np.exp(alpha * dists) / np.max(np.exp(alpha * dists))
        ratios = scaled_rewards / scaled_dists
        correction = traj_samples[np.argmax(ratios)]
        return Choice(selection=correction, options=[correction, t_query])

    def old_correction(self, query: Query) -> Choice:
        curr_state = query.start_state
        feats = [query.task.domain.features(None,curr_state)]
        values = inquire.utils.learning.Learning.discrete_q_iteration(query.task.domain, query.start_state, query.task.get_ground_truth())
        traj = [[None,query.start_state]]
        comp_traj = query.trajectories[0]
        for step in range(len(comp_traj.trajectory)-1):
            opt_action = query.task.domain.avail_actions[np.argmax(values[curr_state[0][0],curr_state[0][1]])]
            comp_state = comp_traj.trajectory[step+1]
            next_state = query.task.domain.next_state(curr_state, opt_action)
            next_action = opt_action
            if comp_traj.trajectory[step][1] == curr_state:
                # Still following proposed trajectory
                comp_x, comp_y = comp_state[1][0]
                comp_val = np.max(values[comp_x,comp_y])
                ## Change to sample using Sampling.weighted_choice
                #opt_prob = (np.max(action_vals) - comp_val) / np.std(action_vals)
                action_vals = values[curr_state[0][0],curr_state[0][1]]

                max_prob = stats.norm.cdf(np.max(action_vals), loc=np.mean(action_vals), scale=np.std(action_vals))
                comp_prob = stats.norm.cdf(comp_val, loc=np.mean(action_vals), scale=np.std(action_vals))
                #if len(accepted_r) == 0 or np.random.rand() < np.mean(r)/(M*np.mean(accepted_r)):
                #if True: #len(accepted_r) == 0 or np.random.rand() > (np.mean(accepted_r) - np.mean(r))/np.mean(accepted_r):
                #if np.random.normal(0,1) >= opt_prob:
                if np.random.rand() >= max_prob - comp_prob:
                    next_state = comp_state[1]
                    next_action = comp_state[0]
            curr_state = next_state
            action = next_action

            traj.append([action,curr_state])
            feats.append(query.task.domain.features(action,curr_state))
        resp = Trajectory(traj, np.sum(feats,axis=0))
        return Choice(resp, [resp] + query.trajectories)

    def binary_feedback(self, query: Query, verbose: bool=False) -> Choice:
        assert(len(query.trajectories) == 1)

        traj_samples = TrajectorySampling.value_sampling(query.start_state, [query.task.get_ground_truth()], query.task.domain, np.random.RandomState(0), self._steps, self._N, {'remove_duplicates': True, 'probabilistic': True})
        # traj_samples = TrajectorySampling.uniform_sampling(query.start_state, None, query.task.domain, np.random.RandomState(0), self._steps, self._N, {'remove_duplicates': True})

        # Construct CDF over rewards
        rewards = np.array([np.dot(t.phi, query.task.get_ground_truth()) for t in traj_samples])
        rewards = np.sort(rewards)
        rewards_cdf = np.linspace(0, 1, self._N)

        # Perform percentile comparison
        percentile_idx = np.argwhere(rewards_cdf >= self._alpha)[0,0]
        threshold_reward = rewards[percentile_idx]
        query_reward = np.dot(query.task.get_ground_truth(), query.trajectories[0].phi)
        bin_fb = query.trajectories[0] if query_reward >= threshold_reward else None

        # Plot CDF
        if verbose:
            print('Reward of query trajectory: {}; Reward of {}th percentile: {}'.format(query_reward, self._alpha*100, threshold_reward))
            print('Plotting CDF over rewards')
            plt.figure()
            plt.plot(rewards, rewards_cdf)
            plt.title('Rewards CDF')
            plt.show()

        return Choice(bin_fb, [query.trajectories[0]])
