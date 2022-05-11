import pdb
import time
import numpy as np

import plotly.express as px

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
                            q = np.max(values[tuple(domain.state_index(next_state))])
                            new_values[state_action_idx] = next_r + (discount * q)
                            delta = np.max([delta, abs(new_values[state_action_idx]-values[state_action_idx])])
                        else:
                            new_values[state_action_idx] = -np.inf
            values = new_values
        return values

    @staticmethod
    def gradient_descent(rand, feedback, gradient_fn, w_dim, sample_count, learning_rate=0.0005, conv_threshold=1.0e-5, viz=True):
        samples = []
        norms = []
        weights_at_lowest = []
        lowest = np.inf
        start = time.perf_counter()
        for _ in range(sample_count):
            init_w = rand.uniform(-1,1,w_dim) #.reshape(-1,1)
            curr_w = init_w/np.linalg.norm(init_w)
            converged = (len(feedback) == 0)

            while not converged:
                grads = gradient_fn(feedback, curr_w)
                new_w = curr_w - (learning_rate * np.array(grads))
                new_w = new_w/np.linalg.norm(new_w)
                norms.append(np.linalg.norm(new_w - curr_w))
                if norms[-1] < lowest:
                    weights_at_lowest = new_w
                    lowest = norms[-1]
                # if np.linalg.norm(new_w - curr_w) < conv_threshold:
                if norms[-1] < conv_threshold:
                    converged = True
                    norms = []
                    curr_w = weights_at_lowest
                    weights_at_lowest = []
                    lowest = np.inf
                    start = time.perf_counter()
                curr_w = new_w
                elapsed = time.perf_counter() - start
                if elapsed >= 90:
                    divisor = 1
                    print("Timeout on gradient descent.")
                    for i in range(2, 1000):
                        if len(norms) % i == 0:
                            divisor = i
                    avg_norms = np.array(norms).reshape(np.max((divisor, int(len(norms)/divisor))), np.min((divisor, int(len(norms)/divisor)))).mean(axis=1)
                    fig = px.line(
                            x=np.arange(avg_norms.shape[0]),
                            y=avg_norms,
                            title=f"{len(norms)} gradient iterations."
                    )
                    fig.show()
                    norms = []
                    curr_w = weights_at_lowest
                    print(f"Lowest norm was: {lowest}\nWeights at that point were: {curr_w}.")
                    weights_at_lowest = []
                    lowest = np.inf
                    start = time.perf_counter()
                    break
            samples.append(curr_w)
        return np.stack(samples)
