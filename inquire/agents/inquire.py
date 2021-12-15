import pdb
import numpy as np
from inquire.interactions.modalities import *
from inquire.interactions.feedback import Query
from inquire.utils.learning import Learning
from inquire.utils.sampling import TrajectorySampling
from inquire.agents.agent import Agent

class Inquire(Agent):
    def __init__(self, sampling_method, sampling_params, M, N, steps, int_types=[]):
        self.M = M # number of weight samples
        self.N = N # number of trajectory samples
        self.steps = steps # trajectory length
        self.int_types = int_types #[Sort, Demo] #, Pref, Rating]
        self.sampling_method = sampling_method
        self.sampling_params = sampling_params

    def reset(self):
        self.rand = np.random.RandomState(0)

    @staticmethod
    def gradient(feedback, w):
        grads = np.zeros_like(w)
        for fb in feedback:
            phis = np.array([f.phi for f in fb.options])
            exps = np.exp(np.dot(phis,w)).reshape(-1,1)
            grads = grads + (fb.selection.phi - np.sum(np.multiply(exps,phis),axis=0)/np.sum(exps))
        return grads * -1

    def generate_exp_mat(self, w_samples, trajectories):
        phi = np.stack([t.phi for t in trajectories])
        exp = np.exp(np.dot(phi, w_samples.T)) # produces a M X N matrix
        exp_mat = np.broadcast_to(exp,(exp.shape[0],exp.shape[0],exp.shape[1]))
        return exp_mat

    def generate_prob_mat(self, exp, int_type): #|Q| x |C| x |W|
        if int_type is Demonstration:
            return np.expand_dims(exp[0] / np.sum(exp, axis=1), axis=0), [list(range(exp.shape[0]))]
        elif int_type is Preference: 
            mat = exp / (exp + np.transpose(exp,(1,0,2)))
            idxs = np.triu_indices(exp.shape[0], 1)
            prob_mat = np.stack([mat[idxs],mat[idxs[::-1]]],axis=1)
            choices = np.transpose(np.stack(idxs))
            return prob_mat, choices
        elif int_type is Correction:
            trans_mat = np.transpose(exp,(1,0,2))
            den_mat = exp + trans_mat
            return exp / den_mat, [[i] for i in range(exp.shape[0])]
        else:
            return None

    def generate_gains_mat(self, prob_mat):
        return prob_mat * np.log(self.M * prob_mat / np.expand_dims(np.sum(prob_mat,axis=-1),axis=-1)) / self.M

    def generate_query(self, domain, query_state, curr_w, verbose=False):
        all_queries, all_gains = [], []
        if verbose:
            print("Sampling trajectories...")
        sampling_params = tuple([query_state, curr_w, domain, self.rand, self.steps, self.N]) + self.sampling_params
        traj_samples = self.sampling_method(*sampling_params)
        exp_mat = self.generate_exp_mat(curr_w, traj_samples)
        for i in self.int_types:
            if verbose:
                print("Assessing " + str(i.__name__) + " queries...")
            prob_mat, choice_idxs = self.generate_prob_mat(exp_mat, i)
            gains = self.generate_gains_mat(prob_mat)
            query_gains = np.sum(gains, axis=(1,2))
            all_gains.append(query_gains)
            all_queries.append(choice_idxs)
        opt_type = np.argmax([np.max(i) for i in all_gains])
        opt_query_idx = np.argmax(all_gains[opt_type])
        query_trajs = [traj_samples[i] for i in all_queries[opt_type][opt_query_idx]]
        opt_query = Query(self.int_types[opt_type], None, query_state, query_trajs)
        return opt_query

    def update_weights(self, domain, feedback):
        return Learning.gradient_descent(self.rand, feedback, Inquire.gradient, domain.w_dim, self.M)
