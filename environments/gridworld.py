from sampling import Sampling
from scipy.stats import norm
import pdb
import itertools
import numpy as np

class GridWorld:
    def __init__(self, random_seed, grid_dim, max_duration, num_puddles, null_action=False):
        #old ground truth weights: x_to_goal, y_to_goal, in_puddle
        #ground truth weights: step cost, at goal, in puddle
        self._r = np.array([-0.5, 1.0, -4.0])
        self._r = self._r / np.linalg.norm(self._r)
        self.max_duration = max_duration
        self.num_puddles = num_puddles
        self.grid_dim = grid_dim
        self.min_dist = grid_dim # minimum distance (manhattan) between start and goal positions in auto-generated states
        self.w_dim = len(self._r)
        #self.avail_actions = list(itertools.product([-1,0,1],repeat=2))
        #self.avail_actions = list(itertools.product([-1,1],[0])) + list(itertools.product([0],[-1,1])) 
        if null_action:
            self.avail_actions = list(itertools.product([-1,1],[0])) + list(itertools.product([0],[-1,1])) + [(0,0)]
        else:
            self.avail_actions = list(itertools.product([-1,1],[0])) + list(itertools.product([0],[-1,1]))
        self.default_state = self.gen_default_state()
        self.values = None
        self.rand = np.random.RandomState(random_seed)

    #def feature_scaling(self, features, max_duration):
    #    return np.array([features[0]/max_duration, features[1], features[2]/max_duration])

    def feature_similarity(self, f1):
        f2 = f1.reshape(f1.shape[0], 1, f1.shape[1])
        dist = np.einsum('ijk, ijk->ij', f1-f2, f1-f2)
        balanced_dists = np.ravel(np.stack([dist,-1*dist]))
        mu, std = norm.fit(balanced_dists)
        scaled_prob = np.zeros((f1.shape[0], f1.shape[0]))
        for i in range(f1.shape[0]):
            for j in range(f1.shape[0]):
                scaled_prob[i,j] = norm.pdf(dist[i,j], mu, std)/norm.pdf(0, mu, std)
        return scaled_prob

    def actions(self, state):
        #puddle_map = state[2]
        #if puddle_map[state[0][0],state[0][1]] > 0:
        #    return [[0,0]] #stuck in puddle
        ref_actions = []
        for a in self.avail_actions:
            new_x = state[0][0] + a[0]
            new_y = state[0][1] + a[1]
            if new_x in range(self.grid_dim) and new_y in range(self.grid_dim):
                ref_actions.append(a)
        return ref_actions

    def next_state(self, state, action):
        ## State rep: [[curr_x,curr_y],[goal_x, goal_y], puddles(NxN)]
        if action in self.actions(state):
            return [[state[0][0] + action[0], state[0][1] + action[1]],state[1],state[2]]
        pdb.set_trace()
        return state #invalid action

    '''def phi(self, traj): #feature trace
        trace = []
        for t in traj:
            trace.append(self.features(t[0],t[1]))
            pdb.set_trace()
        return np.sum(trace,axis=0)'''

    def features(self, action, state): 
        ## State rep: [[curr_x,curr_y],[goal_x, goal_y], puddles(NxN)]
        curr_x, curr_y = state[0]
        goal_x, goal_y = state[1]
        puddle_map = state[2]
        x_to_goal = np.abs(curr_x-goal_x)/self.grid_dim
        y_to_goal = np.abs(curr_y-goal_y)/self.grid_dim
        if action is None or action == (0,0):
            step = 0.0
        else:
            step = 1.0
        in_puddle = puddle_map[curr_x,curr_y]
        at_goal = 1.0 if state[0] == state[1] else 0.0
        return np.array([step/self.max_duration, at_goal, in_puddle/self.max_duration])
        #return np.array([x_to_goal, y_to_goal, in_puddle])

    def trace_reward_truth(self, phi):
        return np.dot(self._r, phi) 

    def w_error(self, w):
        #w_norm = w / np.linalg.norm(w) #np.sum(w**2)
        return np.linalg.norm(self._r - w)
        #return np.sum((self._r - w)**2.0)

    def gen_default_state(self):
        puddle_map = np.zeros((self.grid_dim, self.grid_dim))
        puddle_map[1,4] = 1
        puddle_map[4,6] = 1
        puddle_map[6,2] = 1
        state = [[6,0],[0,6],puddle_map]
        return state

    def gen_state(self):
        ## State rep: [[curr_x,curr_y],[goal_x, goal_y], puddles(NxN)]
        puddle_map = np.zeros((self.grid_dim, self.grid_dim))
        occupied = np.zeros((self.grid_dim, self.grid_dim))
        start_x = self.rand.randint(0,self.grid_dim)
        start_y = self.rand.randint(0,self.grid_dim)
        occupied[start_x,start_y] = 1
        while True:
            goal_x = self.rand.randint(0,self.grid_dim)
            goal_y = self.rand.randint(0,self.grid_dim)
            dist = abs(goal_x - start_x) + abs(goal_y - start_y)
            if dist >= self.min_dist and occupied[goal_x,goal_y] == 0:
                occupied[goal_x,goal_y] = 1
                break
        for i in range(self.num_puddles):
            while True:
                px = self.rand.randint(0,self.grid_dim)
                py = self.rand.randint(0,self.grid_dim)
                if occupied[px,py] == 0:
                    occupied[px,py] = 1
                    puddle_map[px,py] = 1.0
                    break
        return [[start_x,start_y],[goal_x,goal_y],puddle_map]

    def display_traj(self, traj):
        print("TRAJECTORY")
        for pt in traj.traj:
            self.print_state(pt[1])

    def print_state(self, state):
        str_map = "|" + ("---" * self.grid_dim) + "|\n|"
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                char = "   "
                if state[0] == [i,j]:
                    char = " X "
                if state[1] == [i,j]:
                    char = " G "
                if state[2][i,j] == 1:
                    char = " O "
                str_map += char
            str_map += "|\n|"
        str_map += "---" * self.grid_dim + "|"
        print(str_map)

    def print_state_arr(self, states):
        str_map = ("|" + ("---" * self.grid_dim) + "|") * len(states)
        str_map += "\n|"
        for i in range(self.grid_dim):
            for state in states:
                for j in range(self.grid_dim):
                    char = "   "
                    if state[0] == [i,j]:
                        char = " X "
                    if state[1] == [i,j]:
                        char = " G "
                    if state[2][i,j] == 1:
                        char = " O "
                    str_map += char
                str_map += "|    |"
            str_map += "|\n|"
        str_map += ("---" * self.grid_dim + "|     ") * len(states)
        print(str_map)

    def policy(self, state, values):
        actions = self.actions(state)
        action_vals = []
        for a in actions:
            new_state = self.next_state(state, a)
            v = values[new_state[0][0],new_state[0][1]]
            action_vals.append(v)
        return actions, action_vals #actions[np.argmax(action_vals)], actions

    '''def policy_rollout_nondeterministic(self, values, state, steps, t=0.8):
        curr_state = state
        feats = [self.features(curr_state)]
        traj = [[None, state]]
        for step in range(steps):
            actions, action_vals = self.policy(curr_state, values)
            if len(actions) == 1:
                probs = [1.0]
            else:
                probs = [(1.0-t)/float(len(actions)-1)] * len(actions)
                probs[np.argmax(action_vals)] = t
            action = actions[Sampling.weighted_choice(probs)]
            curr_state = self.next_state(curr_state, action)
            traj.append([action,curr_state])
            feats.append(self.features(curr_state))
        return traj, feats #self.trace_reward_truth(np.sum(feats,axis=0))'''

    def policy_rollout(self, values, state, steps):
        curr_state = state
        feats = [self.features(None,curr_state)]
        traj = [[None, state]]
        for step in range(steps):
            #actions, action_vals = self.policy(curr_state, values)
            #opt_action = actions[np.argmax(action_vals)]
            opt_action = self.avail_actions[np.argmax(values[curr_state[0][0],curr_state[0][1]])]
            curr_state = self.next_state(curr_state, opt_action)
            traj.append([opt_action,curr_state])
            feats.append(self.features(opt_action,curr_state))
            if curr_state[0] == curr_state[1]: # Check if terminal state
                break
        return traj, feats #self.trace_reward_truth(np.sum(feats,axis=0))

    def optimal_traj_true_reward(self, state, w, steps, flag=False):
        trajs, rewards = [], []
        values = self.q_iter(state, w, verbose=False, flag=False)
        #if not (w == self._r).all():
        #    pdb.set_trace()
        #values = self.value_iter(state, w, verbose=False)
        traj, feats = self.policy_rollout(values, state, steps)
        reward = self.trace_reward_truth(np.sum(feats,axis=0))
        if flag:
            pdb.set_trace()
        return traj, reward

    def optimal_traj_true_reward_nondeterministic(self, state, w, steps, k=100):
        trajs, rewards = [], []
        values = self.q_iter(state, w, verbose=False)
        #values = self.value_iter(state, w, verbose=False)
        for _ in range(k):
            traj, feats = self.policy_rollout(values, state, steps)
            trajs.append(traj)
            rewards.append(self.trace_reward_truth(np.sum(feats,axis=0)))
        pdb.set_trace()
        return trajs, np.mean(rewards)

    def q_iter_truth(self, state, flag=False,  verbose=True):
        return self.q_iter(state, self._r, flag=flag, verbose=verbose)

    def value_iter_truth(self, state, flag=False,  verbose=True):
        return self.value_iter(state, self._r, flag=flag, verbose=verbose)

    def value_iter(self, state, w, threshold=0.0001, discount=0.9, flag=False, verbose=True):
        values = np.zeros((self.grid_dim, self.grid_dim))
        diff = np.inf
        if verbose:
            print("Running value iteration")
        while diff > threshold:
            diff = 0
            new_values = np.zeros((self.grid_dim, self.grid_dim))
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    action_vals = []
                    curr = [[i,j],state[1],state[2]]
                    curr_r = np.dot(w,self.features(curr))
                    for a in self.actions(curr):
                        next_state = self.next_state(curr, a)
                        action_vals.append(curr_r + (discount * values[next_state[0][0],next_state[0][1]]))
                    diff = np.max([diff, np.abs(values[i,j] - np.max(action_vals))])
                    new_values[i,j] = np.max(action_vals) 
            values = new_values
        if flag:
            pdb.set_trace()
        return values

    def q_iter(self, state, w, threshold=1e-4, discount=0.99, flag=False, verbose=True):
        values = np.zeros((self.grid_dim, self.grid_dim, len(self.avail_actions)))
        diff = np.inf
        terminal_states = [state[1]]
        if verbose:
            print("Running value iteration")
        while diff > threshold:
            diff = 0
            new_values = np.zeros((self.grid_dim, self.grid_dim, len(self.avail_actions)))
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    if [i,j] not in terminal_states:  # Value of terminal state (goal) is 0
                        curr = [[i,j],state[1],state[2]]
                        for a in range(len(self.avail_actions)):
                            #curr_r = np.dot(w,self.features(curr))
                            action = self.avail_actions[a]
                            if action in self.actions(curr):
                                next_state = self.next_state(curr, action)
                                next_r = np.dot(w,self.features(action,next_state))
                                q = np.max(values[next_state[0][0],next_state[0][1]])
                                new_values[i,j,a] = next_r + (discount * q)
                                diff = np.max([diff, abs(new_values[i,j,a]-values[i,j,a])])
                            else:
                                new_values[i,j,a] = -np.inf
            values = new_values
        if flag:
            pdb.set_trace()
        return values

    def value_iter_nondeterministic(self, state, w, threshold=0.01, discount=0.9, t=0.8, verbose=True):
        values = np.zeros((self.grid_dim, self.grid_dim))
        diff = -1
        if verbose:
            print("Running value iteration")
        while diff < 0 or diff > threshold:
            diff = -1
            new_values = np.zeros((self.grid_dim, self.grid_dim))
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    curr = values[i,j]
                    curr_state = [[i,j],state[1],state[2]]
                    #actions = self.actions(curr_state)
                    action_vals = []
                    for a in self.avail_actions:
                        val = 0
                        if a == (0,0):
                            val = np.dot(w, self.features(curr_state)) + (discount * values[curr_state[0][0],curr_state[0][1]])
                        else:
                            state_probs = dict()
                            states = []
                            for ai in self.avail_actions:
                                prob = t if a == ai else (1.0-t)/(len(self.avail_actions)-1)
                                next_state = self.next_state(curr_state, ai)
                                if str(next_state[0]) in state_probs:
                                    state_probs[str(next_state[0])] += prob
                                else:
                                    state_probs[str(next_state[0])] = prob
                                    states.append(next_state)
                            for state in states:
                                val += state_probs[str(state[0])] * (np.dot(w,self.features(state)) + (discount * values[state[0][0],state[0][1]]))
                        action_vals.append(val)
                    diff = np.max([diff, np.abs(curr-np.max(action_vals))])
                    new_values[i,j] = np.max(action_vals) 
            values = new_values
        return values
