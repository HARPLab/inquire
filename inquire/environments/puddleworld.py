from inquire.utils.learning import Learning
from inquire.environments.environment import Environment
from inquire.utils.datatypes import Trajectory
import dtw
import itertools
import numpy as np


class PuddleWorld(Environment):
    def __init__(self, grid_dim, max_duration, num_puddles, null_action=False):
        self.max_duration = max_duration
        self.num_puddles = num_puddles
        self.grid_dim = grid_dim
        self.min_dist = grid_dim # minimum distance (manhattan) between start and goal positions in auto-generated states
        self.w_dim = 3
        #self.avail_actions = list(itertools.product([-1,0,1],repeat=2))
        #self.avail_actions = list(itertools.product([-1,1],[0])) + list(itertools.product([0],[-1,1])) 
        if null_action:
            self.avail_actions = list(itertools.product([-1,1],[0])) + list(itertools.product([0],[-1,1])) + [(0,0)]
        else:
            self.avail_actions = list(itertools.product([-1,1],[0])) + list(itertools.product([0],[-1,1]))

    def maximum_unique_phis(self):
        # Reached goal: (max - min number of steps) * 
        # Number of steps * (reached/didn't reach goal) * number of puddles
        return max_duration * 2 * (num_puddles + 1)

    def generate_random_reward(self, random_state):
        #ground truth weights: step cost, at goal, in puddle
        r = np.array([-0.5, 1.0, -4.0])
        r = r / np.linalg.norm(r)
        return r

    def optimal_trajectory_from_w(self, start_state, w):
        curr_state = start_state
        feats = [self.features(None, curr_state)]
        values = Learning.discrete_q_iteration(self, start_state, w)
        traj = [[None,start_state]]
        for step in range(self.max_duration):
            opt_action = self.avail_actions[np.argmax(values[curr_state[0][0],curr_state[0][1]])]
            curr_state = self.next_state(curr_state, opt_action)
            traj.append([opt_action,curr_state])
            feats.append(self.features(opt_action,curr_state))
            if self.is_terminal_state(curr_state):
                break
        resp = Trajectory(traj, np.sum(feats,axis=0))
        return resp

    def available_actions(self, current_state):
        ref_actions = []
        for a in self.avail_actions:
            new_x = current_state[0][0] + a[0]
            new_y = current_state[0][1] + a[1]
            if new_x in range(self.grid_dim) and new_y in range(self.grid_dim):
                ref_actions.append(a)
        return ref_actions
    
    def sample_action(self, current_state, mean_action, step_size):
        if mean_action is None:
            mean_action = np.array([0,0])
        action_range = np.array([2, 2])
        std_dev = action_range * step_size / 6.0
        sample = np.random.normal(mean_action, std_dev)
        rounded = np.round(sample)
        return tuple([int(i) for i in (mean_action + rounded)])

    def next_state(self, current_state, action):
        ## State rep: [[curr_x,curr_y],[goal_x, goal_y], puddles(NxN)]
        if action in self.available_actions(current_state):
            return [[current_state[0][0] + action[0], current_state[0][1] + action[1]],current_state[1],current_state[2]]
        return current_state #invalid action

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

    def generate_random_state(self, random_state):
        ## State rep: [[curr_x,curr_y],[goal_x, goal_y], puddles(NxN)]
        puddle_map = np.zeros((self.grid_dim, self.grid_dim))
        occupied = np.zeros((self.grid_dim, self.grid_dim))
        start_x = random_state.randint(0,self.grid_dim)
        start_y = random_state.randint(0,self.grid_dim)
        occupied[start_x,start_y] = 1
        while True:
            goal_x = random_state.randint(0,self.grid_dim)
            goal_y = random_state.randint(0,self.grid_dim)
            dist = abs(goal_x - start_x) + abs(goal_y - start_y)
            if dist >= self.min_dist and occupied[goal_x,goal_y] == 0:
                occupied[goal_x,goal_y] = 1
                break
        for i in range(self.num_puddles):
            while True:
                px = random_state.randint(0,self.grid_dim)
                py = random_state.randint(0,self.grid_dim)
                if occupied[px,py] == 0:
                    occupied[px,py] = 1
                    puddle_map[px,py] = 1.0
                    break
        return [[start_x,start_y],[goal_x,goal_y],puddle_map]

    def is_terminal_state(self, current_state):
        return (current_state[0][0] == current_state[1][0]) and (current_state[0][1] == current_state[1][1])

    def all_actions(self):
        return self.avail_actions

    def state_space_dim(self):
        return (self.grid_dim, self.grid_dim)

    def state_space(self, start_state):
        space = []
        goal = start_state[1]
        puddle_map = start_state[2]
        for x in range(self.grid_dim):
            for y in range(self.grid_dim):
                space.append([[x,y],goal,puddle_map])
        return space

    def state_index(self, state):
        return state[0]

    def trajectory_from_states(self, states, features):
        return Trajectory(states, np.sum(features, axis=0))

    def distance_between_trajectories(self, a, b):
        a_points = [state[1][0] for state in a.trajectory]
        b_points = [state[1][0] for state in b.trajectory]
        alignment = dtw.dtw(a_points, b_points)
        return alignment.normalizedDistance
