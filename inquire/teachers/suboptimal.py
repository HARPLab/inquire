from re import sub
from inquire.teachers.teacher import Teacher
from inquire.interactions.feedback import Trajectory, Choice, Query
from inquire.utils.learning import Learning
from inquire.interactions.modalities import Demonstration, Preference, Correction
from inquire.utils.viz import Viz
import numpy as np

class SuboptimalTeacher(Teacher):
    def __init__(self, optimality: float, steps: int) -> None: # , pct: float, N: int, steps: int) -> None:
        super().__init__()
        self.optimality = optimality
        # self.pct = pct
        # self.N = N
        self.steps = steps

    def query(self, q: Query, verbose: bool=False) -> Choice:
        if q.query_type is Demonstration:
            f = self.demonstration(q)
            if verbose:
                print("Showing demonstrated trajectory")
                viz = Viz(f.selection.trajectory)
                while not viz.exit:
                    viz.draw()
            return f
        else:
            raise Exception(self._type.__name__ + " does not support queries of type " + str(q.query_type))

    def demonstration(self, query: Query) -> Choice:
        w = query.task.get_ground_truth()
        domain = query.task.domain
        curr_state = query.start_state
        vals = Learning.discrete_q_iteration(domain, curr_state, w)
        feats = [domain.features(None,curr_state)]
        all_actions = domain.all_actions()
        traj = [[None, curr_state]]
        for _ in range(self.steps):
            action_vals = np.exp(vals[tuple(domain.state_index(curr_state))])
            opt_val_idx = np.argmax(action_vals)
            opt_val = action_vals[opt_val_idx]
            if self.optimality == 1:
                action_vals = np.zeros_like(action_vals)
                action_vals[opt_val_idx] = 1.0
            else:
                scale = self.optimality * (np.sum(action_vals)-opt_val) / (opt_val * (1-self.optimality))
                action_vals[opt_val_idx] *= scale
            if np.sum(action_vals) == 0:
                # Uniform distribution
                action_vals = np.ones_like(action_vals)
            action_vals /= np.sum(action_vals)
            print(action_vals)
            act_idx = np.random.choice(list(range(len(action_vals))), p=action_vals)
            action = all_actions[act_idx]
            new_state = domain.next_state(curr_state, action)
            traj.append([action,new_state])
            feats.append(domain.features(action,new_state))
            curr_state = new_state
            if domain.is_terminal_state(curr_state):
                break
        trajectory = Trajectory(traj, np.sum(feats, axis=0))


        # gt = query.task.get_ground_truth()
        # suboptimal_weights = gt + self.sample_gaussian_noise(gt.shape)
        # suboptimal_weights /= np.linalg.norm(suboptimal_weights)
        # print(gt, suboptimal_weights)
        # traj = query.task.domain.optimal_trajectory_from_noisy_w(query.start_state, suboptimal_weights, 1.0-self.optimality)
        # print('Done query')
        return Choice(trajectory, [trajectory] + query.trajectories)

        # samples = TrajectorySampling.percentile_rejection_sampling(query.start_state, query.task.get_ground_truth(), query.task.domain, np.random, self.steps, self.N, pct=self.pct)
        # traj = np.random.choice(samples) # query.task.domain.optimal_trajectory_from_w(query.start_state, query.task.get_ground_truth())
        # return Choice(traj, [traj] + query.trajectories)
    
    def sample_gaussian_noise(self, size):
        return np.random.normal(loc=0.0, scale=1.0-self.optimality, size=size)

