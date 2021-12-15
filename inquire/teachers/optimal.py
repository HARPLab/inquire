from inquire.interactions.feedback import Trajectory, Choice
from inquire.interactions.modalities import Demonstration, Preference, Correction
import inquire.utils.learning

class OptimalTeacher():

    def query(self, q):
        if q.query_type is Demonstration:
            return self.correction(q)
        elif q.query_type is Preference:
            return self.correction(q)
        elif q.query_type is Correction:
            return self.correction(q)
        else:
            raise Exception(self._type.__name__ + " does not support queries of type " + str(q.query_type))

    def demonstration(self, query):
        traj = query.task.domain.optimal_trajectory_from_w(query.start_state, query.task.get_ground_truth())
        return Choice(resp, [resp] + query.trajectories)

    def preference(self, query):
        r = [np.dot(query.task.get_ground_truth(), qi.phi) for qi in query.trajectories]
        ordering = [qi for _, qi in sorted(zip(r, q))][::-1]
        return Choice(ordering[0], query.trajectories)

    def correction(self, query):
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
