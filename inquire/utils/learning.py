import pdb
import time
import numpy as np
from inquire.utils.sampling import TrajectorySampling
import plotly.express as px

class Learning:

    @staticmethod
    def continuous_policy_iteration(domain, w, init_state, traj_length, threshold=1e-4, discount=0.99):
        states = domain.state_space()
        if states is None:
            samples = 1000
            state_dim = init_state.shape[0]
            low = np.array([np.inf]*state_dim)
            high = np.array([0.0]*state_dim)
            rand = np.random.RandomState(0)
            actions = domain.all_actions()

            random_controls = np.stack([np.random.uniform(low=domain.env.action_space.low[i], high=domain.env.action_space.high[i],size=(samples,traj_length)) for i in range(domain.env.action_space.shape[0])],axis=-1)
            traj_samples = [domain.run(random_controls[c].flatten()) for c in range(samples)]
            low = np.min(np.concatenate([i[0] for i in traj_samples]),axis=0)
            high = np.max(np.concatenate([i[0] for i in traj_samples]),axis=0)
            pdb.set_trace()
            states = []
            for i in range(state_dim.shape[0]):
                states.append(
                    np.linspace(
                        start=low[i], stop=high[i], num=2000, endpoint=True
                    )
                )
            pdb.set_trace()



    @staticmethod
    def discrete_policy_iteration(domain, w, threshold=1e-4, discount=0.99):
        states = domain.state_space()
        pdb.set_trace()

        ## Initialize policy
        policy = dict()
        values = dict()
        for s in states:
            policy[s] = np.random.choice(domain.available_actions(s))
            values[s] = 0

        converged = False
        while not converged:
            ## Policy Evaluation
            value_diff = 0
            while value_diff < threshold:
                value_diff = 0
                for s in states:
                    prev_v = values[s]
                    s_next = domain.next_state(s, policy[s])
                    values[s] = np.dot(domain.features(None, s_next), w) + (discount * values[s_next])
                    value_diff = max(value_diff, abs(values[s] - prev_v))

            ## Policy Improvement
            converged = True
            for s in states:
                prev_s = policy[s]
                actions = domain.available_actions(s)
                neighbors = [domain.next_state(s, a) for a in actions]
                vals = [np.dot(domain.features(None, s_next), w) + (discount * values[s_next]) for s_next in neighbors]
                policy[s] = actions[np.argmax(vals)]
                converged = converged and policy[s] == prev_s

        return policy

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
    def gradient_descent(rand, feedback, gradient_fn, w_dim, sample_count, learning_rate=0.05, conv_threshold=1.0e-5, viz=True):
        samples = []
        for _ in range(sample_count):
            init_w = rand.normal(0,1,w_dim) #.reshape(-1,1)
            curr_w = init_w/np.linalg.norm(init_w)
            converged = (len(feedback) == 0)
            while not converged:
                grads = gradient_fn(feedback, curr_w)
                new_w = curr_w - (learning_rate * np.array(grads))
                new_w = new_w/np.linalg.norm(new_w)
                if np.linalg.norm(new_w - curr_w) < conv_threshold:
                    converged = True
                curr_w = new_w
            samples.append(curr_w)
        return np.stack(samples)

    @staticmethod
    def gradient_descent_with_timeout(rand, feedback, gradient_fn, w_dim, sample_count, learning_rate=0.005, conv_threshold=1.0e-6, viz=True):
        print("Computing the gradient.")
        samples = []
        norms = []
        weights_at_lowest = []
        lowest = np.inf
        start = time.perf_counter()
        for _ in range(sample_count):
            init_w = rand.normal(0,1,w_dim) #.reshape(-1,1)
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
