"""
An agent which uses demonstrations and preferences.

Code adapted from Learning Reward Functions
by Integrating Human Demonstrations and Preferences.
"""
import itertools
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pymc as mc
import scipy.optimize as opt
import theano as th
import theano.tensor as tt

from inquire.agents.agent import Agent
from inquire.environments.environment import Environment
from inquire.interactions.feedback import Choice, Trajectory


class DemPref(Agent):
    """A preference-querying agent seeded with demonstrations.

    Note: We instantiate the agent according to arguments corresponding to
    what the the original paper's codebase designates as their main experiment.
    """

    def __init__(
        self,
        sampling_method,
        optional_sampling_params,
        weight_sample_count: int,
        trajectory_sample_count: int,
        trajectory_length: int,
        interaction_types: list = [],
    ):
        """Initialize the agent."""
        self._sampling_method = sampling_method
        self._optional_sampling_params = optional_sampling_params
        self._weight_sample_count = weight_sample_count
        self._trajectory_sample_count = trajectory_sample_count
        self._trajectory_length = trajectory_length
        self._interaction_types = interaction_types

        """
        Get the pre-defined agent parameters
        """
        self._dempref_agent_parameters = pd.read_csv(
            "dempref_agent_parameters.csv"
        ).to_numpy()
        self._dempref_agent_parameters = self._dempref_agent_parameters[:, 1:]

        """
        Instance attributes from orginal codebase's 'runner.py' object
        """

        self.domain = self._dempref_agent_parameters["domain"]
        self.human_type = self._dempref_agent_parameters["human_type"]

        self.n_demos = self._dempref_agent_parameters["n_demos"]
        self.gen_demos = self._dempref_agent_parameters["gen_demos"]
        self.sim_iter_count = self._dempref_agent_parameters["sim_iter_count"]
        if self.n_demos and not self.gen_demos:
            self.demos = demos[: self.n_demos]
        self.trim_start = self._dempref_agent_parameters["trim_start"]

        self.n_query = self._dempref_agent_parameters["n_query"]
        self.update_func = self._dempref_agent_parameters["update_func"]
        self.query_length = self._dempref_agent_parameters["query_length"]
        self.inc_prev_query = self._dempref_agent_parameters["inc_prev_query"]
        self.gen_scenario = self._dempref_agent_parameters["gen_scenario"]
        self.n_pref_iters = self._dempref_agent_parameters["n_pref_iters"]
        self.epsilon = self._dempref_agent_parameters["epsilon"]

        """
        Instantiate the DemPref-specific sampler and query generator:
        """
        self._query_generator = DemPrefQueryGenerator(
                dom=self.domain,
                num_queries=self.n_query,
                query_length=self.query_length,
                num_expectation_samples=self.n_samples_exp,
                include_previous_query=self.inc_prev_query,
                generate_scenario=self.gen_scenario,
                update_func=self.update_func,
                beta_pref=self.beta_pref,
            )

        self._sampler = DemPrefSampler(
                n_query=self.n_query,
                dim_features=self.domain.feature_size,
                update_func=self.update_func,
                beta_demo=self.beta_demo,
                beta_pref=self.beta_pref,
            )

        assert (
            self.update_func == "pick_best"
            or self.update_func == "approx"
            or self.update_func == "rank"
        ), ("Update" " function must be one of the provided options")
        if self.inc_prev_query and self.human_type == "term":
            assert (
                self.n_demos > 0
            ), "Cannot include previous query if no demonstration is provided"

        self.n_samples_summ = n_samples_summ
        self.n_samples_exp = n_samples_exp

        self.true_weight = true_weight
        self.beta_demo = beta_demo
        self.beta_pref = beta_pref
        self.beta_human = beta_human

        self.config = [
            self.human_type,
            self.n_demos,
            self.trim_start,
            self.n_query,
            self.update_func,
            self.query_length,
            self.inc_prev_query,
            self.gen_scenario,
            self.n_pref_iters,
            self.epsilon,
            self.n_samples_summ,
            self.n_samples_exp,
            self.true_weight,
            self.beta_demo,
            self.beta_pref,
            self.beta_human,
        ]

    def reset(self):
        """Prepare for new query session."""
        self._sampler.clear_pref()
        if self.inc_prev_query and self.n_demos > 0:
            last_query_picked = [d for d in cleaned_demos]

    def generate_query(
        self,
        domain: Environment,
        query_state: np.ndarray,
        curr_w: np.ndarray,
        verbose: bool = False,
    ) -> list:
        """Generate query using approximate gradients.

        Code adapted from DemPref's ApproxQueryGenerator.
        """
        pass

    def update_weights(self, domain: Environment, feedback: Choice):
        """Update the model's learned weights."""
        pass

    def approx_volume_removal(self) -> None:
        """Volume removal objective function."""
        pass

    class DemPrefSampler:
        """Sample trajectories for querying.

        Code adapted from original DemPref agent.
        """

        def __init__(
            self,
            n_query: int,
            dim_features: int,
            update_func: str = "pick_best",
            beta_demo: float = 0.1,
            beta_pref: float = 1.0,
        ):
            """
            Initialize the sampler.

            :param n_query: Number of queries.
            :param dim_features: Dimension of feature vectors.
            :param update_func: options are "rank", "pick_best", and
                                "approx". To use "approx", n_query must be 2.
                                Will throw an assertion error otherwise.
            :param beta_demo: parameter measuring irrationality of human in
                              providing demonstrations
            :param beta_pref: parameter measuring irrationality of human in
                              selecting preferences
            """
            self.n_query = n_query
            self.dim_features = dim_features
            self.update_func = update_func
            self.beta_demo = beta_demo
            self.beta_pref = beta_pref

            if self.update_func == "approx":
                assert (
                    self.n_query == 2
                ), "Cannot use approximation to update function if n_query > 2"
            elif not (
                self.update_func == "rank" or self.update_func == "pick_best"
            ):
                raise Exception(
                    update_func + " is not a valid update function."
                )

            # feature vectors from demonstrated trajectories
            self.phi_demos = np.zeros((1, self.dim_features))
            # a list of np.arrays containing feature difference vectors and
            # which encode the ranking from the preference
            # queries
            self.phi_prefs = []

            self.f = None

        def load_demo(self, phi_demos: np.ndarray):
            """
            Load the demonstrations into the Sampler.

            :param demos: a Numpy array containing feature vectors for each
                          demonstration; has dimension
                          n_dem -by- self.dim_features
            """
            self.phi_demos = phi_demos

        def load_prefs(self, phi: Dict, rank):
            """
            Load the results of a preference query into the Sampler.

            :param phi: a dictionary mapping rankings (0,...,n_query-1) to
                        feature vectors
            """
            result = []
            if self.update_func == "rank":
                result = [None] * len(rank)
                for i in range(len(rank)):
                    result[i] = phi[rank[i]]
            elif self.update_func == "approx":
                result = phi[rank] - phi[1 - rank]
            elif self.update_func == "pick_best":
                result, tmp = [phi[rank] - phi[rank]], []
                for key in sorted(phi.keys()):
                    if key != rank:
                        tmp.append(phi[key] - phi[rank])
                result.extend(tmp)
            self.phi_prefs.append(np.array(result))

        def clear_pref(self):
            """Clear all preference information from the sampler."""
            self.phi_prefs = []

        def sample(self, N: int, T: int = 1, burn: int = 1000) -> List:
            """Return N samples from the distribution.

            The distribution is defined by applying update_func on the
            demonstrations and preferences observed thus far.

            :param N: number of samples to draw.
            :param T: if greater than 1, all samples except each T^{th}
                      sample are discarded
            :param burn: how many samples before the chain converges;
                         these initial samples are discarded
            :return: list of samples drawn
            """
            x = tt.vector()
            x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)

            # define update function
            start = time.time()
            if self.update_func == "approx":
                self.f = th.function(
                    [x],
                    tt.sum(
                        [
                            -tt.nnet.relu(
                                -self.beta_pref * tt.dot(self.phi_prefs[i], x)
                            )
                            for i in range(len(self.phi_prefs))
                        ]
                    )
                    + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)),
                )
            elif self.update_func == "pick_best":
                self.f = th.function(
                    [x],
                    tt.sum(
                        [
                            -tt.log(
                                tt.sum(
                                    tt.exp(
                                        self.beta_pref
                                        * tt.dot(self.phi_prefs[i], x)
                                    )
                                )
                            )
                            for i in range(len(self.phi_prefs))
                        ]
                    )
                    + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)),
                )
            elif self.update_func == "rank":
                self.f = th.function(
                    [x],
                    tt.sum(  # summing across different queries
                        [
                            tt.sum(  # summing across different terms in PL-update
                                -tt.log(
                                    [
                                        tt.sum(  # summing down different feature-differences in a single term in PL-update
                                            tt.exp(
                                                self.beta_pref
                                                * tt.dot(
                                                    self.phi_prefs[i][j:, :]
                                                    - self.phi_prefs[i][j],
                                                    x,
                                                )
                                            )
                                        )
                                        for j in range(self.n_query)
                                    ]
                                )
                            )
                            for i in range(len(self.phi_prefs))
                        ]
                    )
                    + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)),
                )
            print(
                "Finished constructing sampling function in "
                + str(time.time() - start)
                + "seconds"
            )

            # perform sampling
            x = mc.Uniform(
                "x",
                -np.ones(self.dim_features),
                np.ones(self.dim_features),
                value=np.zeros(self.dim_features),
            )

            def sphere(x):
                if (x ** 2).sum() >= 1.0:
                    return -np.inf
                else:
                    return self.f(x)

            # Pat's NOTE: Potential is a "potential term" defined as an "additional
            # tensor...to be added to the model logp" (pymc3 developer guide)

            p = mc.Potential(
                logp=sphere,  # logp stands for log probability
                name="sphere",
                parents={"x": x},
                doc="Sphere potential",
                verbose=0,
            )
            chain = mc.MCMC([x])
            chain.use_step_method(
                mc.AdaptiveMetropolis,
                x,
                delay=burn,
                cov=np.eye(self.dim_features) / 5000,
            )
            chain.sample(N * T + burn, thin=T, burn=burn, verbose=-1)
            samples = x.trace()
            samples = np.array([x / np.linalg.norm(x) for x in samples])

            # print("Finished MCMC after drawing " + str(N*T+burn) + " samples")
            return samples

    class DemPrefQueryGenerator:
        """Generate queries.

        Code adapted from original DemPref agent.
        """

        def __init__(
            self,
            # dom: domain.Domain,
            env: Environment,
            num_queries: int,
            query_length: int,
            num_expectation_samples: int,
            include_previous_query: bool,
            generate_scenario: bool,
            update_func: str,
            beta_pref: float,
        ) -> None:
            """
            Initialize the approx query generation.

            Note: this class generates queries using approx gradients.

            :param dom: the domain to generate queries on
            :param num_queries: number of queries to generate at each time step
            :param query_length: the length of each query
            :param num_expectation_samples: number of samples to use in
                                            approximating the objective
                                            function
            :param include_previous_query: boolean for whether one of the
                                           queries is the previously selected
                                           query
            :param generate_scenario: boolean for whether we want to generate
                                      the scenario -- i.e., other agents'
                                      behavior
            :param update_func: the update_func used; the options are
                                "pick_best", "approx", and "rank"
            :param beta_pref: the rationality parameter for the human
                              selecting her query
            """
            assert (
                num_queries >= 1
            ), "QueryGenerator.__init__: num_queries must be at least 1"
            assert (
                query_length >= 1
            ), "QueryGenerator.__init__: query_length must be at least 1"
            assert (
                num_expectation_samples >= 1
            ), "QueryGenerator.__init__: num_expectation_samples must be \
                    at least 1"
            # self.domain = dom
            self.env = env
            self.num_queries = num_queries
            self.query_length = query_length
            self.num_expectation_samples = num_expectation_samples
            self.include_previous_query = include_previous_query
            self.generate_scenario = (
                generate_scenario  # Currently must be False
            )
            assert (
                self.generate_scenario is False
            ), "Cannot generate scenario when using approximate gradients"
            self.update_func = update_func
            self.beta_pref = beta_pref
            self.num_new_queries = (
                self.num_queries - 1
                if self.include_previous_query
                else self.num_queries
            )

        def queries(
            self,
            w_samples: np.ndarray,
            last_query: Trajectory = None,
            blank_traj: bool = False,
        ) -> List[Trajectory]:
            """
            Generate self.num_queries number of queries.

            This function produces queries that (locally) maximize the maximum
            volume removal objective.

            :param w_samples: Samples of w
            :param last_query: The previously selected query. Only required if
                               self.inc_prev_query is True
            :param blank_traj: True is last_query is blank. (Only True if not
                               using Dempref but using inc_prev_)
            :return: a list of trajectories (queries)
            """
            start = time.time()

            def func(controls: np.ndarray, *args) -> float:
                """Minimize via L_BFGS.

                :param controls: an array, concatenated to contain the control
                                 input for all queries
                :param args: the first argument is the domain, and the second
                             is the samples that will be used to approximate
                             the objective function
                :return: the value of the objective function for the given set
                         of controls
                """
                # domain = args[0]
                env = args[0]
                samples = args[1]
                # features = generate_features(env, controls, last_query)
                features = env.features(controls, last_query)
                if self.update_func == "pick_best":
                    return -objective(features, samples)
                elif self.update_func == "approx":
                    return -approx_objective(features, samples)
                else:
                    return -rank_objective(features, samples)

            # def generate_features(
            #    domain: domain.Domain,
            #    controls: np.ndarray,
            #    last_query: Trajectory = None,
            # ) -> List:
            #    """
            #    Generates a set of features for the set of controls provided.

            #    :param domain: the domain that the queries are being generated on.
            #    :param controls: an array, concatenated to contain the control input for all queries.
            #    :param last_query: the last query chosen by the human. Only required if self.inc_prev_query is true.
            #    :return: a list containing the feature values of each query.
            #    """
            #    z = self.query_length * domain.control_size
            #    controls = np.array(controls)
            #    controls_set = [
            #        controls[i * z : (i + 1) * z]
            #        for i in range(self.num_new_queries)
            #    ]
            #    trajs = [domain.run(c) for c in controls_set]
            #    features = [domain.np_features(traj) for traj in trajs]
            #    if self.include_previous_query and not blank_traj:

            #        features.append(domain.np_features(last_query))
            #    return features

            def objective(features: List, samples: np.ndarray) -> float:
                """
                The standard maximum volume removal objective function.

                :param features: a list containing the feature values of each
                                 query
                :param samples: samples of w, used to approximate the objective
                :return: the value of the objective function, evaluated on the
                         given queries' features
                """
                volumes_removed = []
                for i in range(len(features)):
                    feature_diff = np.array(
                        [f - features[i] for f in features]
                    )  # n_queries x feature_size
                    weighted_feature_diff = (
                        np.sum(np.dot(feature_diff, samples.T), axis=1)
                        / samples.shape[0]
                    )  # n_queries x 1 -- summed across samples
                    v_removed = 1.0 - 1.0 / np.sum(
                        np.exp(self.beta_pref * weighted_feature_diff)
                    )
                    volumes_removed.append(v_removed)
                return np.min(volumes_removed)

            def approx_objective(features, samples) -> float:
                """
                The approximate maximum volume removal objective function.

                :param features: a list containing the feature values of each
                                 query
                :param samples: samples of w, used to approximate the objective
                :return: the value of the objective function, evaluated on the
                         given queries' features
                """
                volumes_removed = []
                for i in range(len(features)):
                    feature_diff = (
                        features[i] - features[1 - i]
                    )  # 1 x feature_size
                    weighted_feature_diff = (
                        np.sum(np.dot(feature_diff, samples.T))
                        / samples.shape[0]
                    )  # 1 x 1 -- summed across samples
                    v_removed = 1.0 - np.minimum(
                        1.0, np.exp(self.beta_pref * weighted_feature_diff)
                    )
                    volumes_removed.append(v_removed)
                return np.min(volumes_removed)

            def rank_objective(features, samples) -> float:
                """
                The ranking maximum volume removal objective function.

                Note: This objective uses the Plackett-Luce model of
                human behavior.

                CANNOT BE USED WITH (INC_PREV_QUERY AND NO DEMPREF).

                :param features: a list containing the feature values of each
                                 query
                :param samples: samples of w, used to approximate the objective
                :return: the value of the objective function, evaluated on the
                         given queries' features
                """
                # features: n_queries x feature_size
                # samples: n_samples x feature_size
                exp_rewards = (
                    np.sum(np.dot(features, samples.T), axis=1)
                    / samples.shape[0]
                )  # n_queries x 1 -- summed across samples
                volumes_removed = []
                rankings = itertools.permutations(
                    list(range(self.num_queries))
                )  # iterating over all possible rankings
                for rank in rankings:
                    exp_rewards_sorted = [None] * len(rank)
                    for i in range(len(rank)):
                        exp_rewards_sorted[rank[i]] = exp_rewards[i]

                    value, i = 1, 0
                    for i in range(len(rank) - 1):
                        value *= 1.0 / np.sum(
                            np.exp(
                                self.beta_pref
                                * (
                                    np.array(exp_rewards_sorted[i:])
                                    - exp_rewards_sorted[i]
                                )
                            )
                        )
                    volumes_removed.append(1 - value)
                return np.min(volumes_removed)

            z = self.query_length * self.domain.control_size
            lower_input_bound = [
                x[0] for x in self.domain.control_bounds
            ] * self.query_length
            upper_input_bound = [
                x[1] for x in self.domain.control_bounds
            ] * self.query_length
            opt_res = opt.fmin_l_bfgs_b(
                func,
                x0=np.random.uniform(
                    low=self.num_new_queries * lower_input_bound,
                    high=self.num_new_queries * upper_input_bound,
                    size=(self.num_new_queries * z),
                ),
                args=(self.domain, w_samples),
                bounds=self.domain.control_bounds
                * self.num_new_queries
                * self.query_length,
                approx_grad=True,
            )
            query_controls = [
                opt_res[0][i * z : (i + 1) * z]
                for i in range(self.num_new_queries)
            ]
            end = time.time()
            print("Finished computing queries in " + str(end - start) + "s")
            if self.include_previous_query and not blank_traj:
                return [last_query] + [
                    self.domain.run(c) for c in query_controls
                ]
            else:
                return [self.domain.run(c) for c in query_controls]
