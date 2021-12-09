from sampling import Sampling
from viz import Viz
import time
import pdb
from feedback import Traj
import numpy as np
from gridworld import GridWorld

class GridTestSet():
    def __init__(self, K, steps):
        random_seed = 101
        self.domain = GridWorld(random_seed, 8, steps, 12)
        self.cases = [self.domain.gen_state() for _ in range(K)]
        self.steps = steps
        self.max_t = [self.domain.optimal_traj_true_reward(s, self.domain._r, steps) for s in self.cases]
        self.max_traj = [t[0] for t in self.max_t]
        self.max_r = [t[1] for t in self.max_t]

    def eval(self, w):
        perf = [self.domain.optimal_traj_true_reward(c, w, self.steps) for c in self.cases]
        trajs = [t[0] for t in perf]
        rs = [t[1] for t in perf]
        frac = []
        for ri in range(len(self.cases)):
            if self.max_r[ri] > 0:
                frac.append(rs[ri]/self.max_r[ri])
            else:
                pdb.set_trace()
        exp_frac = [np.exp(rs[ri])/np.exp(self.max_r[ri]) for ri in range(len(self.cases))]
        #frac = [rs[ri]/self.max_r[ri] for ri in range(len(self.cases))]
        sub = [(rs[ri] - self.max_r[ri]) / self.max_r[ri] for ri in range(len(self.cases))]
        if np.max(frac) > 1.0:
            pdb.set_trace()
        return np.mean(frac)
        #return np.sum([self.domain.optimal_traj_true_reward(c, w, self.steps) for c in self.cases]) / float(len(self.cases))

class Session():
    def __init__(self, M, N, steps, random_seed=5, int_types=[]):
        self.M = M # number of weight samples
        self.N = N # number of trajectory samples
        self.steps = steps # trajectory length
        self.domain = GridWorld(random_seed, 8, steps, 12)
        self.int_types = int_types #[Sort, Demo] #, Pref, Rating]
        self.rand = np.random.RandomState(random_seed)

    def main(self, test_set, query_count=50, sample_type=None, fixed_state=False, random_query=False, display_traj=False, verbose=True):
        metrics, all_feedback = [], []
        it = 0
        w_samples = Sampling.optimize_w(self.rand, self.domain, all_feedback, self.M)
        #w_samples = Sampling.sample_w(self.rand, self.domain, all_feedback, self.M, step_size=0.5)
        s = self.domain.default_state
        #state_rand = self.rand #np.random.RandomState(0)

        while it < query_count:
            gains, all_queries = [], []
            t0 = time.time()

            ## Get next state
            if not fixed_state:
                s = self.domain.gen_state() #state_rand)

            ## Evaluate info gain for each interaction type
            for i in self.int_types:
                query_gains = []
                if verbose:
                    print("Sampling trajectories...")
                if sample_type is None:
                    traj_samples = i.sample_trajs(s, self.domain, w_samples, self.rand, self.N, self.steps)
                elif sample_type == "uniform":
                    traj_samples = Sampling.uniform_traj_sample(s, self.domain, self.rand, self.N, self.steps)
                elif sample_type == "rejection1":
                    traj_samples = Sampling.rejection_traj_sample_percentile(s, self.domain, w_samples, self.rand, self.N, self.steps)
                elif sample_type.startswith("rejection"):
                    threshold = float(sample_type.split("rejection")[1])
                    print("threshold: " + str(threshold))
                    traj_samples = Sampling.rejection_traj_sample(s, self.domain, w_samples, self.rand, self.N, self.steps, threshold=threshold)
                elif sample_type == "value-det":
                    traj_samples = Sampling.value_traj_sample(s, self.domain, w_samples, self.rand, self.N, self.steps, probabilistic=False, remove_duplicates=False)
                elif sample_type == "value-prob":
                    traj_samples = Sampling.value_traj_sample(s, self.domain, w_samples, self.rand, self.N, self.steps, probabilistic=True, remove_duplicates=False)
                else:
                    print("Sample type " + str(sample_type) + " unkown")
                    exit()

                ## Calculate expected reward for all trajectory/weight pairings
                if verbose:
                    print("Evaluating " + str(i.__name__) + " queries...")
                #phi = np.stack([t.phi for t in traj_samples])
                #exp = np.exp(np.dot(phi, w_samples.T)) # produces a M X N matrix
                #sim = self.domain.feature_similarity(phi)

                #if verbose:
                #    print("Getting probability matrix...")
                #prob_mat1 = i.get_prob_mat(exp)
                #prob_mat = i.get_sim_weighted_prob_mat(exp, sim)

                #prob_mat1 = Sampling.prob_mat(traj_samples, w_samples, dist_weighted=False)
                #prob_mat = Sampling.prob_mat(traj_samples, w_samples, dist_weighted=True)
                #prob_mat2 = Sampling.prob_mat(traj_samples_dup, w_samples, dist_weighted=False)
                prob_mat = Sampling.prob_mat(traj_samples, w_samples, dist_weighted=False)
                #prob_mat4 = Sampling.prob_mat(traj_samples+[traj_samples[0]], w_samples, dist_weighted=False)
                #prob_mat5 = Sampling.prob_mat(traj_samples+[traj_samples[0]], w_samples, dist_weighted=True)

                ## Calculate info gain over all queries
                if verbose:
                    print("Calculating gains...")
                all_gains = (1.0/self.M) * np.sum(prob_mat.T * np.log(self.M * prob_mat.T / np.sum(prob_mat,axis=-1).T),axis=0)
                gains.append(i.sum_over_choices(all_gains))
                all_queries.append(traj_samples)

            ## Select best query
            if verbose:
                print("Selecting query...")
            if random_query:
                best_type = self.rand.randint(0,len(all_queries))
                q_idx = [np.unravel_index(self.rand.randint(0,np.size(g)),g.shape) for g in gains]
                best_idx = (best_type, q_idx[best_type])
                #best_idx = (best_type, self.rand.randint(0,len(all_queries[best_type])))
            else:
                max_idxs = [np.unravel_index(np.argmax(g),g.shape) for g in gains]
                best_type = np.argmax([gains[i][max_idxs[i]] for i in range(len(gains))])
                best_idx = (best_type, max_idxs[best_type]) #returns (int_type, query_num)
                pdb.set_trace()

            ## Pose best query to oracle
            if verbose:
                print("Posing query to oracle...")
            query_type = self.int_types[best_idx[0]]
            query_trajs = all_queries[best_idx[0]] #[best_idx[1]]
            resp = query_type.query_oracle(self.domain, self.steps, query_trajs, best_idx[1], display=display_traj)
            all_feedback.append(resp)
            #viz = Viz(resp[0].traj)
            #while not viz.exit:
            #    viz.draw()

            ## Report metrics
            if verbose:
                print("Resampling w...")
            w_samples = Sampling.optimize_w(self.rand, self.domain, all_feedback, self.M)
            #w_samples = Sampling.sample_w(self.rand, self.domain, all_feedback, self.M)
            if verbose:
                print("Done resampling w...")
            w_mean = np.mean(w_samples,axis=0)
            norm_mean = w_mean #/ np.linalg.norm(w_mean)
            w_var = np.var(w_samples,axis=0)
            perf = test_set.eval(norm_mean)
            error = self.domain.w_error(norm_mean)
            if perf > 1:
                pdb.set_trace()
            print("Query #" + str(it+1) + "; " + query_type.__name__)
            print("elapsed: " + str.format('{0:.3f}', time.time()-t0)) # + ", sample time: " + str.format('{0:.3f}', t1-t0))
            print("  perf: " + str(perf))
            print(" error: " + str(error))
            print("mean_w: " + str(norm_mean))
            print(" var_w: " + str(w_var) + '\n')
            metrics.append([error,perf]) #[error, norm_mean, w_var])
            it += 1
        return metrics
