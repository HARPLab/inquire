import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import norm
import scipy.stats
from feedback import Traj
import numpy as np
import pdb
import math
import random

class Sampling:

    @staticmethod
    def weighted_choice(weights):
        normed = weights/np.sum(weights)
        pdb.set_trace()
        r = np.random.uniform(0,np.sum(weights))
        val = 0.0
        for wi in range(len(weights)):
            val += weights[wi] 
            if val > r:
                return wi
        return -1

    @staticmethod
    def value_traj_sample(state, domain, w_samples, rand, N, steps, probabilistic=False, remove_duplicates=False):
        trajs = []
        values = [domain.q_iter(state, wi, verbose=False) for wi in w_samples]
        init_feats = domain.features(None,state)
        
        for i in range(N):
            vals = values[rand.randint(0,len(w_samples))]
            curr_state = state
            feats = [init_feats]
            traj = [[None,state]]
            for step in range(steps):
                actions = domain.avail_actions #actions(curr_state)
                action_vals = np.exp(vals[curr_state[0][0],curr_state[0][1]])
                #for a in actions:
                #    new_state = domain.next_state(curr_state, a)
                #    v = vals[new_state[0][0],new_state[0][1]]
                #    action_vals.append(np.exp(v))
                if probabilistic:
                    choice_weights = action_vals/np.sum(action_vals)
                    act_idx = np.random.choice(list(range(len(action_vals))), p=choice_weights)
                else:
                    act_idx = np.argmax(action_vals)
                action = actions[act_idx]
                curr_state = domain.next_state(curr_state, action)
                traj.append([action,curr_state])
                feats.append(domain.features(action,curr_state))
                if curr_state[0] == curr_state[1]: # Check if terminal state
                    break

            if remove_duplicates:
                duplicate = False
                new_traj = Traj(traj, np.sum(feats,axis=0))
                for t in trajs:
                    if (new_traj.phi == t.phi).all():
                        duplicate = True
                        break
                if not duplicate:
                    trajs.append(new_traj)
            else:
                trajs.append(Traj(traj, np.sum(feats,axis=0)))
        return trajs

    @staticmethod
    def uniform_traj_sample(state, domain, rand, N, steps):
        samples = []
        for i in range(N):
            curr_state = state
            traj = [[None, curr_state]]
            feats = [domain.features(None,curr_state)]
            for j in range(steps):
                actions = domain.actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if curr_state[0] == curr_state[1]: # Check if terminal state
                    break
            samples.append(Traj(traj, np.sum(feats,axis=0)))
        return samples

    @staticmethod
    def rejection_traj_sample_percentile(state, domain, w_samples, rand, N, steps, pct=0.8):
        sample_size = math.ceil(float(N) / (1.0-pct))
        samples, rewards = [], []
        for _ in range(sample_size):
            curr_state = state
            traj = [[None, curr_state]]
            feats = [domain.features(None,curr_state)]
            for j in range(steps):
                actions = domain.actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if curr_state[0] == curr_state[1]: # Check if terminal state
                    break

                phi = np.sum(feats,axis=0)
                new_traj = Traj(traj, phi)
                samples.append(new_traj)
                r = [np.dot(new_traj.phi,wi) for wi in w_samples]
                rewards.append(r)
        var = np.var(rewards,axis=1)
        mean = np.mean(rewards,axis=1)
        rang = np.ptp(rewards,axis=1)
        idxs = np.argsort(var)[::-1]
        top_samples = [samples[idxs[i]] for i in range(N)]
        return top_samples

    @staticmethod
    def rejection_traj_sample(state, domain, w_samples, rand, N, steps, threshold=0.1, burn=1000):
        samples, accepted_r = [], []
        accepted_flag, accepted_all, accepted_full = [],[], []
        while len(accepted_flag) < (burn + N):
            curr_state = state
            traj = [[None, curr_state]]
            feats = [domain.features(None,curr_state)]
            for j in range(steps):
                actions = domain.actions(curr_state)
                ax = rand.randint(0,len(actions))
                new_state = domain.next_state(curr_state, actions[ax])
                traj.append([actions[ax],new_state])
                feats.append(domain.features(actions[ax],new_state))
                curr_state = new_state
                if curr_state[0] == curr_state[1]: # Check if terminal state
                    break

                phi = np.sum(feats,axis=0)
                new_traj = Traj(traj, phi)
                r = [np.dot(new_traj.phi,wi) for wi in w_samples]

                if len(accepted_r) < 2 or np.std(accepted_r) == 0:
                    opt_prob = np.inf 
                else:
                    #try max instead of np.mean(r) next
                    opt_prob = scipy.stats.norm.cdf(np.mean(r), loc=np.mean(accepted_r), scale=np.std(accepted_r))
                if threshold < 0:
                    flag = np.random.rand() < opt_prob
                else:
                    flag = opt_prob > 0 and np.log(np.random.rand()) < np.log(opt_prob) - np.log(threshold)
                if flag:
                    samples.append(new_traj)
                    accepted_r.append(np.mean(r))
                    accepted_all.append(r)
            if flag:
                accepted_flag.append(r)
            accepted_full.append(r)
        metrics_all = [np.mean(np.median(accepted_all,axis=1)), np.mean(np.var(accepted_all,axis=1)), np.mean(np.ptp(accepted_all,axis=1))]
        metrics_full = [np.mean(np.median(accepted_full,axis=1)), np.mean(np.var(accepted_full,axis=1)), np.mean(np.ptp(accepted_full,axis=1))]
        if len(accepted_flag) > 0:
            metrics_flag = [np.mean(np.median(accepted_flag,axis=1)), np.mean(np.var(accepted_flag,axis=1)), np.mean(np.ptp(accepted_flag,axis=1))]
        pdb.set_trace()
        return samples[burn:]

    @staticmethod
    def rejection_traj_sample1(state, domain, w_samples, rand, N, steps):
        samples = []
        accepted_r=[]
        for i in range(N):
            accepted = False
            while not accepted:
                curr_state = state
                traj = [[None, curr_state]]
                feats = [domain.features(None,curr_state)]
                for j in range(steps):
                    actions = domain.actions(curr_state)
                    ax = rand.randint(0,len(actions))
                    new_state = domain.next_state(curr_state, actions[ax])
                    traj.append([actions[ax],new_state])
                    feats.append(domain.features(actions[ax],new_state))
                    curr_state = new_state
                    if curr_state[0] == curr_state[1]: # Check if terminal state
                        break
                phi = np.sum(feats,axis=0)
                new_traj = Traj(traj, phi)
                r = [np.dot(new_traj.phi,wi) for wi in w_samples]

                if len(accepted_r) < 2 or np.std(accepted_r) == 0:
                    opt_prob = np.inf 
                else:
                    opt_prob = scipy.stats.norm.cdf(np.mean(r), loc=np.mean(accepted_r), scale=np.std(accepted_r))
                #if len(accepted_r) == 0 or np.random.rand() < np.mean(r)/(M*np.mean(accepted_r)):
                #if True: #len(accepted_r) == 0 or np.random.rand() > (np.mean(accepted_r) - np.mean(r))/np.mean(accepted_r):
                #if np.random.normal(0,1) >= opt_prob:
                if np.random.rand() < opt_prob:
                    samples.append(new_traj)
                    accepted_r.append(np.mean(r))
                    accepted = True
        return samples

    @staticmethod
    def feature_similarity(f1):
        f2 = f1.reshape(f1.shape[0], 1, f1.shape[1])
        dist = np.einsum('ijk, ijk->ij', f1-f2, f1-f2)
        balanced_dists = np.ravel(np.stack([dist,-1*dist]))
        mu, std = norm.fit(balanced_dists)
        scaled_prob = np.zeros((f1.shape[0], f1.shape[0]))
        for i in range(f1.shape[0]):
            for j in range(f1.shape[0]):
                scaled_prob[i,j] = norm.pdf(dist[i,j], mu, std)/norm.pdf(0, mu, std)
        return scaled_prob

    @staticmethod
    def sim_weighted_prob_mat(exp,sim):
        num = (exp.T / np.sum(sim, axis=0)).T
        den = np.sum(num,axis=0)
        return num/den

    @staticmethod
    def unweighted_prob_mat(exp):
        prob_mat = exp / np.sum(exp,axis=0)
        return prob_mat

    @staticmethod
    def prob_mat(choices, w_samples, dist_weighted=False):
        phi = np.stack([t.phi for t in choices])
        exp = np.exp(np.dot(phi, w_samples.T)) # produces a M X N matrix
        if dist_weighted:
            sim = Sampling.feature_similarity(phi)
            prob_mat = Sampling.sim_weighted_prob_mat(exp, sim)
        else:
            prob_mat = Sampling.unweighted_prob_mat(exp)
        return prob_mat

    @staticmethod
    def optimize_w(rand, domain, feedback, sample_count, learning_rate=0.05, conv_threshold=1.0e-5, viz=True):
        samples = []
        grad_samples = []

        for _ in range(sample_count):
            init_w = rand.uniform(-1,1,domain.w_dim) #.reshape(-1,1)
            curr_w = init_w/np.linalg.norm(init_w)
            init = True
            #curr_w = init_w
            converged = False

            while not converged:
                grads = [0.0]*domain.w_dim #.reshape(-1,1)
                for fb in feedback:
                    phis = np.array([f.phi for f in fb])
                    exps = np.exp(np.dot(phis,curr_w)).reshape(-1,1)
                    grads = grads + (fb[0].phi - np.sum(np.multiply(exps,phis),axis=0)/np.sum(exps))
                grad_samples.append(list(curr_w) + [np.linalg.norm(grads)])
                new_w = curr_w + (learning_rate * np.array(grads))
                new_w = new_w/np.linalg.norm(new_w)
                if np.linalg.norm(new_w - curr_w) < conv_threshold:
                    converged = True
                curr_w = new_w
            samples.append(new_w)
            #comp = Sampling.slow_opt(domain, feedback, np.squeeze(init_w))
            #print("diff: " + str(np.linalg.norm(samples[-1]-comp)))
        if viz and len(feedback) > 0:
            ax = plt.axes(projection='3d')
            datapoints = np.stack(grad_samples)
            #ax.plot_trisurf(datapoints[:,0], datapoints[:,1], datapoints[:,2], c=datapoints[:,3], cmap='viridis', linewidth=0.5)
            ax.scatter(datapoints[:,0], datapoints[:,1], datapoints[:,2], c=datapoints[:,3], cmap='viridis', linewidth=0.5)
            plt.show()

        return np.stack(samples)

    @staticmethod
    def slow_opt(domain, feedback, init_w, learning_rate=0.05, conv_threshold=1.0e-5):
        curr_w = init_w
        converged = False

        while not converged:
            new_w = [0]*domain.w_dim # np.array([0]*domain.w_dim) #.reshape(-1,1)
            for j in range(init_w.shape[0]):
                grad = 0.0
                for fb in feedback:
                    opt_phi = fb[0].phi[j]
                    num = 0.0
                    den = 0.0
                    for t in fb:
                        phi = t.phi
                        num = num + (phi[j] * np.exp(np.dot(phi,curr_w)))
                        den = den + np.exp(np.dot(phi,curr_w))
                    grad = grad + (opt_phi - (num/den))
                new_w[j] = curr_w[j] + (learning_rate * grad)
            new_w = new_w / np.linalg.norm(new_w)
            diff = np.abs(curr_w - new_w)
            if np.max(diff) < conv_threshold:
                converged = True
            curr_w = new_w
        return new_w

    @staticmethod
    def logprob(w, feedback):
        if np.sum(w**2) > 1:
            return -np.inf

        probs = []
        for fb in feedback:
            #phi = [c.phi for c in fb]
            #exp = np.exp(np.dot(phi,w))
            #probs.append((exp[0]/np.sum(exp)))
            #phi = [c.phi for c in fb]
            #exp = np.exp(np.dot(phi,w))
            #prob = (exp[0]/np.sum(exp))
            prob_mat = Sampling.prob_mat(fb, np.expand_dims(w,axis=0), dist_weighted=False) #True)
            probs.append(prob_mat[0] / np.sum(prob_mat))
        #return np.log(np.prod(probs))
        #    probs.append(np.log(exp[0]/np.sum(exp)))
        return np.prod(np.array(probs))
        #return np.sum(np.array([np.log(p) for p in probs]))

    @staticmethod
    def sample_w(rand, domain, feedback, sample_count, burn=1000, thin=50, step_size=0.1):
        ## Metropolis-Hastings sampling from easy-active-learning
        x = np.array([0]*domain.w_dim).reshape(1,-1)
        old_logprob = Sampling.logprob(x[0], feedback)
        for i in range(burn + thin*sample_count):
            new_x = x[-1] + (rand.normal(0,1,domain.w_dim) * step_size)
            #new_x = new_x / np.linalg.norm(new_x)
            new_logprob = Sampling.logprob(new_x, feedback)
            p = rand.rand()
            #print([p, new_logprob / old_logprob])
            if p < new_logprob / old_logprob:
                #if len(feedback) > 0 and i > burn:
                #    print([new_x, new_logprob, old_logprob])
                x = np.vstack((x,new_x))
                old_logprob = new_logprob
            else:
                x = np.vstack((x,x[-1]))
        x = x[burn+thin-1::thin]
        '''if len(feedback) > 0:
            w = domain._r
            mat1 = Sampling.prob_mat(feedback[0], np.expand_dims(w,axis=0), dist_weighted=False) #True)
            mat2 = Sampling.prob_mat(feedback[0], np.expand_dims(w,axis=0), dist_weighted=True)
            pdb.set_trace()'''
        return x

