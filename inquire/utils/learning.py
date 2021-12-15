import numpy as np
import random

class Learning:
    @staticmethod
    def discrete_q_iteration(domain, start_state, w, threshold=1e-4, discount=0.99):
        state_space_dim = domain.state_space_dim()
        state_space = domain.state_space(start_state)
        values_dim = tuple([i for i in state_space_dim] + [len(domain.all_actions())])
        all_actions = domain.all_actions()
        values = np.zeros(values_dim)
        delta = np.inf
        while delta > threshold:
            delta = 0
            new_values = np.zeros(values_dim)
            for state in state_space:
                if not domain.is_terminal_state(state):  # Value of terminal state (goal) is 0
                    state_idx = tuple(domain.state_index(state))
                    for a in range(len(all_actions)):
                        state_action_idx = state_idx + tuple([a])
                        action = all_actions[a]
                        if action in domain.available_actions(state):
                            next_state = domain.next_state(state, action)
                            next_r = np.dot(w,domain.features(action,next_state))
                            q = np.max(values[domain.state_index(next_state)])
                            new_values[state_action_idx] = next_r + (discount * q)
                            delta = np.max([delta, abs(new_values[state_action_idx]-values[state_action_idx])])
                        else:
                            new_values[state_action_idx] = -np.inf
            values = new_values
        return values

    @staticmethod
    def gradient_descent(rand, feedback, gradient_fn, sample_count, learning_rate=0.05, conv_threshold=1.0e-5, viz=True):
        samples = []

        for _ in range(sample_count):
            init_w = rand.uniform(-1,1,domain.w_dim) #.reshape(-1,1)
            curr_w = init_w/np.linalg.norm(init_w)
            converged = False

            while not converged:
                grads = gradient_fn(feedback, curr_w)
                new_w = curr_w - (learning_rate * np.array(grads))
                new_w = new_w/np.linalg.norm(new_w)
                if np.linalg.norm(new_w - curr_w) < conv_threshold:
                    converged = True
                curr_w = new_w
            samples.append(new_w)

        return np.stack(samples)
