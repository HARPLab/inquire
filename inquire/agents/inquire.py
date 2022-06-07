import scipy
import pdb
import numpy as np
import pandas as pd
from inquire.utils.datatypes import Query, Feedback, Choice, Modality
from inquire.utils.learning import Learning
from inquire.utils.sampling import TrajectorySampling
from inquire.agents.agent import Agent
#import matplotlib.pyplot as plt

class FixedInteractions(Agent):
    def __init__(self, sampling_method, optional_sampling_params, M, N, steps, int_types=[]):
        self.M = M # number of weight samples
        self.N = N # number of trajectory samples
        self.steps = steps # trajectory length
        self.int_types = int_types #[Sort, Demo] #, Pref, Rating]
        self.sampling_method = sampling_method
        self.optional_sampling_params = optional_sampling_params
        self.query_num = 0

    def initialize_weights(self, rand, domain):
        init_w = rand.normal(0,1,(domain.w_dim(), self.M)) #.reshape(-1,1)
        init_w = init_w/np.linalg.norm(init_w, axis=0)
        return init_w.T

    def reset(self):
        self.rand = np.random.RandomState(0)
        self.query_num = 0

    def generate_query(self, domain, query_state, curr_w, verbose=False):
        all_queries, all_gains = [], []
        if verbose:
            print("Sampling trajectories...")
        if isinstance(start_state, CachedSamples):
            traj_samples = self.rand.choice(start_state.traj_samples, self.N)
        else:
            sampling_params = tuple([query_state, curr_w, domain, self.rand, self.steps, self.N, self.optional_sampling_params])
            traj_samples = self.sampling_method(*sampling_params)
        exp_mat = Inquire.generate_exp_mat(curr_w, traj_samples)

        i = self.int_types[self.query_num]
        if verbose:
            print("Assessing " + str(i.name) + " queries...")
        prob_mat, choice_idxs = Inquire.generate_prob_mat(exp_mat, i)
        gains = Inquire.generate_gains_mat(prob_mat, self.M)
        query_gains = np.sum(gains, axis=(1,2))

        opt_query_idx = np.argmax(query_gains)
        query_trajs = [traj_samples[a] for a in choice_idxs[opt_query_idx]]
        opt_query = Query(i, None, query_state, query_trajs)
        self.query_num += 1
        return opt_query

    def step_weights(self, curr_w, domain, feedback):
        converted_feedback = self.convert_binary_feedback_to_prefs(curr_w, feedback, domain)
        return Learning.gradient_descent(self.rand, converted_feedback, Inquire.gradient, domain.w_dim(), self.M, conv_threshold=np.inf)

    def update_weights(self, curr_w, domain, feedback):
        converted_feedback = self.convert_binary_feedback_to_prefs(curr_w, feedback, domain)
        return Learning.gradient_descent(self.rand, converted_feedback, Inquire.gradient, domain.w_dim(), self.M)

class Inquire(Agent):
    def __init__(self, sampling_method, optional_sampling_params, M, N, steps, int_types=[], beta=10.0):
        self.M = M # number of weight samples
        self.N = N # number of trajectory samples
        self.steps = steps # trajectory length
        self.int_types = int_types #[Sort, Demo] #, Pref, Rating]
        self.sampling_method = sampling_method
        self.optional_sampling_params = optional_sampling_params
        self.chosen_interactions = []
        self.beta = beta

    @staticmethod
    def scale_reward(r, reward_range):
        # Scales to [0,1]
        return (r - reward_range.min) / (reward_range.max - reward_range.min)

    def reset(self):
        self.rand = np.random.RandomState(0)

    def initialize_weights(self, rand, domain):
        init_w = rand.normal(0,1,(domain.w_dim(), self.M)) #.reshape(-1,1)
        init_w = init_w/np.linalg.norm(init_w, axis=0)
        return init_w.T

    @staticmethod
    def pairwise_gradient(feedback, w):
        grads = np.zeros_like(w)
        for fb in feedback:
            phi_pos = fb.choice.selection.phi
            for f in fb.choice.options:
                if any(f.phi != phi_pos):
                    phis = np.array([f.phi, phi_pos])
                    exps = np.exp(np.dot(phis,w)).reshape(-1,1)
                    grads = grads + (phi_pos - ((exps*phis).sum(axis=0)/exps.sum()))
        return grads * -1

    @staticmethod
    def gradient(feedback, w, beta):
        grads = np.zeros_like(w)
        for fb in feedback:
            phi_pos = fb.choice.selection.phi
            phis = np.array([f.phi for f in fb.choice.options])
            unique_phis = np.unique(phis, axis=0)
            exps = np.exp(beta*np.dot(unique_phis,w)).reshape(-1,1)
            grads = grads + ((beta * phi_pos) - ((beta*exps*unique_phis).sum(axis=0)/exps.sum()))
        return grads * -1

    @staticmethod
    def generate_exp_mat(w_samples, trajectories, beta):
        phi = np.stack([t.phi for t in trajectories])
        exp = np.exp(beta * np.dot(phi, w_samples.T)) # produces a M X N matrix
        #plt.tight_layout()
        #plt.hist(exp.flatten(), bins=100)
        #plt.show()
        exp_mat = np.broadcast_to(exp,(exp.shape[0],exp.shape[0],exp.shape[1]))
        return exp_mat

    @staticmethod
    def generate_prob_mat(exp, int_type): #|Q| x |C| x |W|
        mat = exp / (exp + np.transpose(exp,(1,0,2)))
        diag = np.repeat(np.expand_dims(np.eye(mat.shape[0], mat.shape[1], dtype=bool), axis=-1), mat.shape[-1], axis=-1)
        if int_type is Modality.DEMONSTRATION:
            choice_matrix = np.expand_dims(np.array(list(range(exp.shape[0]))),axis=0)
            return np.expand_dims(exp[0] / np.sum(exp, axis=1), axis=0), choice_matrix
        elif int_type is Modality.DEMONSTRATION_PAIRWISE:
            choice_matrix = np.expand_dims(np.array(list(range(exp.shape[0]))),axis=0)
            prod_mat = np.prod(mat, axis=1) / mat[0,0]
            return np.expand_dims(prod_mat/np.sum(prod_mat,axis=0), axis=0), choice_matrix
        elif int_type is Modality.PREFERENCE:  
            idxs = np.triu_indices(exp.shape[0], 1)
            prob_mat = np.stack([mat[idxs],mat[idxs[::-1]]],axis=1)
            choices = np.transpose(np.stack(idxs))
            return prob_mat, choices
        elif int_type is Modality.CORRECTION:
            tf_mat = np.transpose(mat, (1,0,2))
            result = np.transpose(tf_mat/np.sum(tf_mat,axis=0),(1,0,2)), [[i] for i in range(exp.shape[0])]
            return result
        elif int_type is Modality.BINARY:
            choice_matrix = np.expand_dims(np.array(list(range(exp.shape[0]))),axis=1)
            pref_mat = (np.sum(mat,axis=0)-0.5)/(mat.shape[0]-1)
            return np.stack([pref_mat, 1.0-pref_mat],axis=1), choice_matrix
        else:
            return None

    @staticmethod
    def generate_gains_mat(prob_mat, M):
        return prob_mat * np.log(M * prob_mat / np.expand_dims(np.sum(prob_mat,axis=-1),axis=-1)) 

    def generate_query(self, domain, query_state, curr_w, verbose=False):
        all_queries, all_gains = [], []
        all_probs = []
        if verbose:
            print("Sampling trajectories...")
        sampling_params = tuple([query_state, curr_w, domain, self.rand, self.steps, self.N, self.optional_sampling_params])
        traj_samples = self.sampling_method(*sampling_params)
        exp_mat = Inquire.generate_exp_mat(curr_w, traj_samples, self.beta)

        for i in self.int_types:
            if verbose:
                print("Assessing " + str(i.name) + " queries...")
            prob_mat, choice_idxs = Inquire.generate_prob_mat(exp_mat, i)
            gains = Inquire.generate_gains_mat(prob_mat, self.M)
            query_gains = np.sum(gains, axis=(1,2)) / self.M
            #query_gains = np.mean(np.sum(gains, axis=-1), axis=-1)
            all_gains.append(query_gains)
            all_queries.append(choice_idxs)
            all_probs.append(prob_mat)
        if verbose:
            print("Selecting best query...")
        gains = [np.max(i) for i in all_gains]
        opt_type = np.argmax([np.max(i) for i in all_gains])
        opt_query_idx = np.argmax(all_gains[opt_type])
        query_trajs = [traj_samples[i] for i in all_queries[opt_type][opt_query_idx]]
        opt_query = Query(self.int_types[opt_type], None, query_state, query_trajs)
        if verbose:
            print(f"Chosen interaction type: {self.int_types[opt_type].name}")
        self.chosen_interactions.append(self.int_types[opt_type].name)
        return opt_query

    @staticmethod
    def convert_binary_feedback(curr_w, feedback, traj_samples):
        converted_feedback = []
        for i in range(len(feedback)):
            fb = feedback[i]
            traj = fb.choice.options[0]
            if fb.modality is Modality.BINARY:
                sign = fb.choice.selection
                rewards = np.array([np.dot(curr_w, t.phi) for t in traj_samples[i]])
                lower_threshold_r = np.percentile(rewards, 25) #replace with whatever percentile threshold
                upper_threshold_r = np.percentile(rewards, 75) #replace with whatever percentile threshold

                for j in range(len(traj_samples[i])):
                    if sign and rewards[j] <= lower_threshold_r:
                        converted_feedback.append(Feedback(Modality.PREFERENCE, Choice(traj, [traj, traj_samples[i][j]])))
                    if (not sign) and rewards[j] >= upper_threshold_r:
                        converted_feedback.append(Feedback(Modality.PREFERENCE, Choice(traj_samples[i][j], [traj, traj_samples[i][j]])))
            else:
                converted_feedback.append(fb)
        return converted_feedback

    def step_weights(self, curr_w, domain, feedback):
        return self.update_weights(None, domain, feedback, conv_threshold=np.inf)

    def update_weights(self, init_w, domain, feedback, momentum = 0.0, learning_rate=0.05, conv_threshold=1.0e-5):
        traj_samples = []
        for fb in feedback:
            if fb.modality is Modality.BINARY:
                traj = fb.choice.options[0]
                query_state = traj.trajectory[0][1]
                sampling_params = tuple([query_state, init_w, domain, self.rand, self.steps, self.N, self.optional_sampling_params])
                traj_samples.append(self.sampling_method(*sampling_params))
            else:
                traj_samples.append(None)

        return Learning.gradient_descent(self.rand, feedback, Inquire.gradient, self.beta, domain.w_dim(), self.M, Inquire.convert_binary_feedback, traj_samples, momentum, learning_rate, conv_threshold)

    def save_data(self, directory: str, file_name: str, data: np.ndarray = None) -> None:
        """Save the agent's stored attributes."""
        if data is not None:
            data = np.stack(data, axis=1).squeeze()
            df = pd.DataFrame(data)
            df.to_csv(directory + file_name)
        else:
            df = pd.DataFrame(self.chosen_interactions)
            df.to_csv(directory + file_name)
