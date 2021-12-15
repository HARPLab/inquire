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
            exps = np.exp(np.dot(phis,curr_w)).reshape(-1,1)
            grads = grads + (fb.selection.phi - np.sum(np.multiply(exps,phis),axis=0)/np.sum(exps))
        return grads * -1

    def generate_exp_mat(self, w_samples, trajectories):
        phi = np.stack([t.phi for t in trajectories])
        exp = np.exp(np.dot(phi, w_samples.T)) # produces a M X N matrix
        exp_mat = np.broadcast_to(exp,(exp.shape[0],exp.shape[0],exp.shape[1]))
        #trans_mat = np.transpose(num_mat,(1,0,2))
        #den_mat = num_mat + trans_mat
        #return trans_mat/den_mat
        return exp_mat

    def generate_prob_mat(self, exp, int_type): #|Q| x |C| x |W|
        if int_type is Demonstration:
            return (exp[0] / np.sum(exp, axis=1)).reshape(1,-1), list(range(exp.shape[0]))
        elif int_type is Preference: #not quite right
            mat = exp / (exp + np.transpose(exp,(1,0,2)))
            up_idxs = np.triu_indices(exp.shape[0])
            lw_idxs = np.tril_indices(exp.shape[0])
            prob_mat = np.stack([mat[up_idxs], mat[lw_idxs]]).reshape(-1,2,exp.shape[-1])
            choices = np.stack([up_idxs, lw_idxs]).reshape(-1,2,exp.shape[-1])
            return prob_mat, choices
        elif int_type is Correction:
            trans_mat = np.transpose(exp,(1,0,2))
            den_mat = exp + trans_mat
            return exp / den_mat, list(range(exp.shape[0]))
        else:
            return None

    def generate_gains_mat(self, prob_mat):
        sum_w = np.broadcast_to(np.sum(prob_mat,axis=-1), prob_mat.shape[0], prob_mat.shape[1], prob_mat.shape[2])
        return prob_mat * np.log(self.M * prob_mat / sum_w) / self.M

    def generate_query(self, domain, query_state, curr_w):
        all_queries, all_gains = [], []
        sampling_params = tuple([query_state, curr_w, domain, self.rand, self.steps, self.N]) + self.sampling_params
        traj_samples = self.sampling_method(sampling_params)
        for i in self.int_types:
            prob_mat = self.generate_prob_mat(traj_samples, w_samples, dist_weighted=False)
            gains, choice_idxs = self.generate_gains_mat(prob_mat)
            query_gains = np.sum(gains, axis=0)
            all_gains.append(query_gains)
            all_queries.append(choice_idxs)
        gains = np.stack(all_gains)
        max_idxs = [np.unravel_index(np.argmax(g),g.shape) for g in gains]
        best_type = np.argmax([gains[i][max_idxs[i]] for i in range(len(gains))])
        return query_trajs[best_type][max_idxs[best_type]]

    def update_weights(self, domain, feedback):
        return Learning.gradient_descent(self.rand, feedback, Inquire.gradient, self.M)
