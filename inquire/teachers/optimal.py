from inquire.teachers.teacher import Teacher
from inquire.utils.datatypes import Choice, Query, Feedback, Modality
from typing import Union
#from inquire.interactions.modalities import Modality.DEMONSTRATION, Modality.PREFERENCE, Modality.CORRECTION, Modality.BINARY
# from inquire.utils.viz import Viz
from inquire.utils.sampling import TrajectorySampling
from inquire.environments.environment import CachedTask, Task
import numpy as np


class OptimalTeacher(Teacher):
    @property
    def alpha(self):
        return self._alpha

    def __init__(self, N, display_interactions: bool = False) -> None:
        super().__init__()
        self._alpha = 0.75
        self._N = N
        self._display_interactions = display_interactions

    def query_response(self, q: Query, task: Union[Task, CachedTask], verbose: bool=False) -> Choice:
        if q.query_type is Modality.DEMONSTRATION:
            f = self.demonstration(q, task)
            if self._display_interactions:
                print("Showing demonstrated trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
            return f
        elif q.query_type is Modality.PREFERENCE:
            f = self.preference(q, task)
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
        elif q.query_type is Modality.CORRECTION:
            f = self.correction(q, task)
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
        elif q.query_type is Modality.BINARY:
            f = self.binary_feedback(q, task)
            if verbose:
                print("Teacher Feedback: {}".format("+1" if f.choice.selection else "-1"))
            if self._display_interactions:
                print("Teacher Feedback: {}".format("+1" if f.choice.selection else "-1"))
                print("Showing original trajectory")
                viz = Viz(q.trajectories[0].trajectory)
                while not viz.exit:
                    viz.draw()
            return f
        else:
            raise Exception(self._type.__name__ + " does not support queries of type " + str(q.query_type))

    def demonstration(self, query: Query, task: Union[Task, CachedTask]) -> Choice:
        traj = task.optimal_trajectory_from_ground_truth(query.start_state)
        return Feedback(Modality.DEMONSTRATION, query, Choice(traj, [traj] + query.trajectories))

    def preference(self, query: Query, task: Union[Task, CachedTask]) -> Choice:
        r = [task.ground_truth_reward(qi) for qi in query.trajectories]
        return Feedback(Modality.PREFERENCE, query, Choice(selection=query.trajectories[np.argmax(r)], options=query.trajectories))

    def correction(self, query: Query, task: Union[Task, CachedTask]) -> Choice:
        t_query = query.trajectories[0]
        min_r = task.ground_truth_reward(t_query)
        opt_traj = task.optimal_trajectory_from_ground_truth(query.start_state)
        if min_r == task.ground_truth_reward(opt_traj):
            return None # Query trajectory is already optimal

        alpha = 1.0 #optional parameter for discouraging more distanced corrections
        samples, rewards, dists = [], [], []
        rand = np.random.RandomState(0)
        while len(rewards) < self._N:
            traj_samples = TrajectorySampling.uniform_sampling(query.start_state, None, task.domain, rand, task.domain.trajectory_length, self._N, {'remove_duplicates': False})
            for t in traj_samples:
                t_reward = task.ground_truth_reward(t)
                if t_reward >= min_r:
                    samples.append(t)
                    rewards.append(t_reward)
                    t_dist = task.domain.distance_between_trajectories(t, t_query)
                    dists.append(t_dist)
        max_r = np.max(np.array(rewards))
        if max_r == min_r: # Could not find a better trajectory than the queried one
            return None
        scaled_rewards = np.array([(r - min_r) / (max_r - min_r) for r in rewards])
        dists = np.array(dists)
        scaled_dists = np.exp(alpha * dists) / np.max(np.exp(alpha * dists))
        ratios = scaled_rewards / scaled_dists
        correction = samples[np.argmax(ratios)]
        return Feedback(Modality.CORRECTION, query, Choice(selection=correction, options=[correction, t_query]))

    def binary_feedback(self, query: Query, task: Union[Task, CachedTask]) -> Choice:
        assert(len(query.trajectories) == 1)

        traj_samples = TrajectorySampling.uniform_sampling(query.start_state, None, task.domain, np.random.RandomState(0), task.domain.trajectory_length, self._N, {'remove_duplicates': False})

        # Construct CDF over rewards
        rewards = np.array([np.dot(t.phi, task.get_ground_truth()) for t in traj_samples])
        rewards = np.sort(rewards)
        rewards_cdf = np.linspace(0, 1, self._N)

        # Perform percentile comparison
        percentile_idx = np.argwhere(rewards_cdf >= self._alpha)[0,0]
        threshold_reward = rewards[percentile_idx]
        query_reward = task.ground_truth_reward(query.trajectories[0])
        bin_fb = (query_reward >= threshold_reward) 

        return Feedback(query.query_type, query, Choice(bin_fb, [query.trajectories[0]]))
