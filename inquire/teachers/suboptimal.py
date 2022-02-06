from re import sub
from inquire.teachers.teacher import Teacher
from inquire.interactions.feedback import Trajectory, Choice, Query
from inquire.utils.learning import Learning
from inquire.interactions.modalities import Demonstration, Preference, Correction
from inquire.utils.viz import Viz
import numpy as np

class SuboptimalTeacher(Teacher):
    def __init__(self, optimality: float, steps: int) -> None:
        super().__init__()
        self.optimality = optimality
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
            action_vals = vals[tuple(domain.state_index(curr_state))]
            original_vals = np.exp(action_vals)/np.sum(np.exp(action_vals))
            opt_val_idx = np.argmax(action_vals)
            opt_val = action_vals[opt_val_idx]
            # TODO handle special case for optimality == 0
            if self.optimality == 1:
                action_vals = np.zeros_like(action_vals)
                action_vals[opt_val_idx] = 1.0
            else:
                action_val_consts = np.where(np.isinf(action_vals), 1,
                    np.log(self.optimality*action_vals / (1-self.optimality)) * (
                        self.optimality*action_vals + (1-self.optimality)*opt_val
                    ) / (
                        0.5*(opt_val+action_vals)
                    )
                )
            # action_val_consts = np.ones_like(action_val_consts)*100 # Starts to make a difference in reward probabilities when constants are ~100
            action_vals = np.exp(action_vals*action_val_consts)
            action_vals /= np.sum(action_vals)
            print(action_vals, original_vals)
            act_idx = np.random.choice(list(range(len(action_vals))), p=action_vals)
            action = all_actions[act_idx]
            new_state = domain.next_state(curr_state, action)
            traj.append([action,new_state])
            feats.append(domain.features(action,new_state))
            curr_state = new_state
            if domain.is_terminal_state(curr_state):
                break
        trajectory = Trajectory(traj, np.sum(feats, axis=0))

        return Choice(trajectory, [trajectory] + query.trajectories)
    
    # def demonstration(self, query: Query) -> Choice:
    #     w = query.task.get_ground_truth()
    #     domain = query.task.domain
    #     curr_state = query.start_state
    #     vals = Learning.discrete_q_iteration(domain, curr_state, w)
    #     feats = [domain.features(None,curr_state)]
    #     all_actions = domain.all_actions()
    #     traj = [[None, curr_state]]
    #     for _ in range(self.steps):
    #         action_vals = np.exp(vals[tuple(domain.state_index(curr_state))])
    #         opt_val_idx = np.argmax(action_vals)
    #         opt_val = action_vals[opt_val_idx]
    #         if self.optimality == 1:
    #             action_vals = np.zeros_like(action_vals)
    #             action_vals[opt_val_idx] = 1.0
    #         else:
    #             scale = self.optimality * (np.sum(action_vals)-opt_val) / (opt_val * (1-self.optimality))
    #             action_vals[opt_val_idx] *= scale
    #         if np.sum(action_vals) == 0:
    #             # Uniform distribution
    #             action_vals = np.ones_like(action_vals)
    #         action_vals /= np.sum(action_vals)
    #         print(action_vals)
    #         act_idx = np.random.choice(list(range(len(action_vals))), p=action_vals)
    #         action = all_actions[act_idx]
    #         new_state = domain.next_state(curr_state, action)
    #         traj.append([action,new_state])
    #         feats.append(domain.features(action,new_state))
    #         curr_state = new_state
    #         if domain.is_terminal_state(curr_state):
    #             break
    #     trajectory = Trajectory(traj, np.sum(feats, axis=0))

    #     return Choice(trajectory, [trajectory] + query.trajectories)
