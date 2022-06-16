"""
Agent that seeds learning w/ demonstrations and then asks preference queries.

Code adapted from Learning Reward Functions
by Integrating Human Demonstrations and Preferences.
"""
import itertools
import time
from typing import Dict, List

import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from inquire.agents.agent import Agent
from inquire.environments.environment import Environment
from inquire.utils.datatypes import Modality, Query, Trajectory
from inquire.utils.sampling import TrajectorySampling


class DemPref(Agent):
    """A preference-querying agent seeded with demonstrations.

    Note: We instantiate the agent according to arguments corresponding to
    what the the original paper's codebase designates as their main experiment.
    """

    def __init__(
        self,
        weight_sample_count: int,
        trajectory_sample_count: int,
        interaction_types: list = [],
        w_dim: int = 4,
        visualize: bool = False,
        seed_with_n_demos: int = 3,
        domain_name: str = None,
        opt_iter_count: int = 1000,
    ):
        """Initialize the agent.

        Note we needn't maintain a domain's start state; that's handled in
        inquire/tests/evaluation.py and the respective domain.
        """
        self._weight_sample_count = weight_sample_count
        self._trajectory_sample_count = trajectory_sample_count
        self._interaction_types = interaction_types
        self._visualize = visualize
        self._human_experiments = True

        """
        Set the agent parameters:
        """
        self._dempref_agent_parameters = {
            "teacher_type": "opt",
            "update_func": "approx",
            "epsilon": 0.0,
            "beta_demo": 0.1,
            "beta_pref": 5,
            "beta_teacher": 1,
            "n_demos": seed_with_n_demos,
            "n_iters_exp": 8,
            "n_pref_iters": 20,
            "n_samples_exp": 50000,
            "n_samples_summ": 2000,
            "query_option_count": 2,
            "opt_iter_count": opt_iter_count,
            "trajectory_length": 10,
            "trim_start": 0,
            "gen_demos": True,
            "gen_scenario": False,
            "incl_prev_query": False,
            "true_weight": [[-0.4, 0.4, -0.2, -0.7]],
        }

        """
        Instance attributes from orginal codebase's 'runner.py' object. Note
        that some variable names are modified to be consist with the Inquire
        parlance.
        """
        self.domain_name = domain_name
        self.rng = np.random.RandomState(0)
        print(f"DemPref agent acting in {self.domain_name} domain.")
        self.teacher_type = self._dempref_agent_parameters["teacher_type"]

        self.n_demos = self._dempref_agent_parameters["n_demos"]
        print(f"DemPref agent seeding with {self.n_demos} demos.")
        self.gen_demos = self._dempref_agent_parameters["gen_demos"]
        self.opt_iter_count = self._dempref_agent_parameters["opt_iter_count"]
        self.trim_start = self._dempref_agent_parameters["trim_start"]

        self.query_option_count = self._dempref_agent_parameters[
            "query_option_count"
        ]
        self.update_func = self._dempref_agent_parameters["update_func"]
        print(f"DemPref agent using {self.update_func} update function.")
        self.trajectory_length = self._dempref_agent_parameters[
            "trajectory_length"
        ]
        print(
            f"DemPref agent considering trajectories of length {self.trajectory_length}."
        )
        self.incl_prev_query = self._dempref_agent_parameters[
            "incl_prev_query"
        ]
        self.gen_scenario = self._dempref_agent_parameters["gen_scenario"]
        self.n_pref_iters = self._dempref_agent_parameters["n_pref_iters"]
        self.epsilon = self._dempref_agent_parameters["epsilon"]

        """
        Instantiate the DemPref-specific sampler and query generator:
        """
        self._sampler = None
        self._w_samples = None
        self._query_generator = None
        self._w_dim = w_dim

        assert (
            self.update_func == "pick_best"
            or self.update_func == "approx"
            or self.update_func == "rank"
        ), ("Update" " function must be one of the provided options")
        if self.incl_prev_query and self.teacher_type == "term":
            assert (
                self.n_demos > 0
            ), "Cannot include previous query if no demonstration is provided"

        self.n_samples_summ = self._dempref_agent_parameters["n_samples_summ"]
        self.n_samples_exp = self._dempref_agent_parameters["n_samples_exp"]
        self.beta_demo = self._dempref_agent_parameters["beta_demo"]
        self.beta_pref = self._dempref_agent_parameters["beta_pref"]
        self.beta_teacher = self._dempref_agent_parameters["beta_teacher"]

    def initialize_weights(
        self, random_number_generator, domain: Environment
    ) -> np.ndarray:
        """Placeholder function."""
        return self.w_samples

    def reset(self) -> None:
        """Prepare for new query session."""
        if self._sampler is not None:
            self._sampler.clear_pref()
        self._sampler = self.DemPrefSampler(
            query_option_count=self.query_option_count,
            dim_features=self._w_dim,
            update_func=self.update_func,
            beta_demo=self.beta_demo,
            beta_pref=self.beta_pref,
            visualize=self._visualize,
        )
        self.w_samples = self._sampler.sample(N=self.n_samples_summ)
        return self.w_samples

    def generate_query(
        self,
        domain: Environment,
        query_state: int,
        curr_w: np.ndarray,
        verbose: bool = False,
    ) -> list:
        """Generate query using approximate gradients.

        Code adapted from DemPref's ApproxQueryGenerator.
        """
        if self._query_generator is None:
            self._query_generator = self.DemPrefQueryGenerator(
                dom=domain,
                num_queries=self.query_option_count,
                num_expectation_samples=self.n_samples_exp,
                include_previous_query=self.incl_prev_query,
                generate_scenario=self.gen_scenario,
                update_func=self.update_func,
                beta_pref=self.beta_pref,
            )
        if self.incl_prev_query:
            if len(self.demos) > 0:
                self.random_scenario_index = np.random.randint(len(self.demos))
            else:
                self.random_scenario_index = 0
            last_query_choice = self.all_query_choices[
                self.random_scenario_index
            ]

        # Generate query_options while ensuring that features of query_options
        # are epsilon apart:
        query_diff = 0
        print("Generating query_options")
        while query_diff <= self.epsilon:
            if self.incl_prev_query:
                if last_query_choice.null:
                    query_options = (
                        self._query_generator.generate_query_options(
                            w_samples=self.w_samples,
                            start_state=query_state,
                            blank_traj=True,
                        )
                    )
                else:
                    query_options = (
                        self._query_generator.generate_query_options(
                            w_samples=self.w_samples,
                            start_state=query_state,
                            last_query_choice=last_query_choice,
                        )
                    )
            else:
                query_options = self._query_generator.generate_query_options(
                    w_samples=self.w_samples, start_state=query_state
                )
            query_diffs = []
            for m in range(len(query_options)):
                for n in range(m):
                    query_diffs.append(
                        np.linalg.norm(
                            query_options[m].phi - query_options[n].phi
                        )
                    )
            query_diff = max(query_diffs)

        query = Query(
            query_type=Modality.PREFERENCE,
            start_state=query_state,
            trajectories=query_options,
        )
        return query

    def generate_demo_query(self, query_state, domain):
        """Request a demonstration with which we'll seed the agent."""
        if domain.__class__.__name__ == "PizzaMaking":
            t_length = np.random.randint(domain._max_topping_count)
        else:
            t_length = self.trajectory_length
        sampled_trajectories = TrajectorySampling.uniform_sampling(
            query_state,
            None,
            domain,
            self.rng,
            t_length,
            self.opt_iter_count,
            {},
        )
        query = Query(
            query_type=Modality.DEMONSTRATION,
            start_state=query_state,
            trajectories=sampled_trajectories,
        )
        return query

    def update_weights(
        self,
        current_weights: np.ndarray,
        domain: Environment,
        feedback: list,
        learning_rate: float,
        sample_threshold: float,
        opt_threshold: float,
    ) -> np.ndarray:
        """Update the model's learned weights.

        ::inputs:
            ::current_weights: Irrelevant for DemPref; useful to other agents
            ::domain: The task's environment
            ::feedback: A list of the human feedback received to this point.
                        DemPref utilizes only the most recent
        """
        if feedback == []:
            # No feedback yet received
            return self.w_samples
        else:
            # Use the most recent Choice in feedback:
            query_options = feedback[-1].choice.options
            choice = feedback[-1].choice.selection
            choice_index = query_options.index(choice)
            if self.incl_prev_query:
                self.all_query_choices[self.random_scenario_index] = choice

            # Create dictionary map from rankings to query-option features;
            # load into sampler:
            features = [x.phi for x in query_options]
            phi = {k: features[k] for k in range(len(query_options))}
            self._sampler.load_prefs(phi, choice_index)
            self.w_samples = self._sampler.sample(N=self.n_samples_summ)
            # Return the new weights from the samples:
            mean_w = np.mean(self.w_samples, axis=0)
            mean_w = mean_w / np.linalg.norm(mean_w)
            mean_w = mean_w.reshape(1, -1)
            return self.w_samples, self.w_samples

    def seed_with_demonstrations(self, feedback: list) -> None:
        """Seed with demonstration."""
        if self.n_demos > 0:
            self.demos = []
            for d in range(len(feedback)):
                self.demos.append(feedback[d].choice.selection)
            phis_from_demos = [x.phi for x in self.demos]
            self._sampler.load_phis_from_demos(np.array(phis_from_demos))
            self.cleaned_demos = self.demos
            if self.incl_prev_query:
                self.all_query_choices = [d for d in self.cleaned_demos]
        self.reset()

    class DemPrefSampler:
        """Sample trajectories for querying.

        Code adapted from original DemPref agent.
        """

        def __init__(
            self,
            query_option_count: int,
            dim_features: int,
            update_func: str = "approx",
            beta_demo: float = 0.1,
            beta_pref: float = 1.0,
            visualize: bool = False,
        ):
            """
            Initialize the sampler.

            :param query_option_count: Number of queries.
            :param dim_features: Dimension of feature vectors.
            :param update_func: options are "rank", "pick_best", and
                                "approx". To use "approx", query_option_count
                                must be 2; will throw an assertion error
                                otherwise
            :param beta_demo: parameter measuring irrationality of teacher in
                              providing demonstrations
            :param beta_pref: parameter measuring irrationality of teacher in
                              selecting preferences
            """
            self.query_option_count = query_option_count
            self.dim_features = dim_features
            self.update_func = update_func
            self.beta_demo = beta_demo
            self.beta_pref = beta_pref
            self._visualize = visualize

            if self.update_func == "approx":
                assert (
                    self.query_option_count == 2
                ), "Cannot use approximation to update function if query_option_count > 2"
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

        def load_phis_from_demos(self, phi_demos: np.ndarray):
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

            :param phi: a dictionary mapping rankings
                        (0,...,query_option_count-1) to feature vectors
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

        def sample(self, N: int, T: int = 1, burn: int = 1000) -> np.ndarray:
            """Return N samples from the distribution.

            The distribution is defined by applying update_func on the
            demonstrations and preferences observed thus far.

            :param N: number of w_samples to draw.
            :param T: if greater than 1, all samples except each T^{th}
                      sample are discarded
            :param burn: how many samples before the chain converges;
                         these initial samples are discarded
            :return: list of w_samples drawn

            NOTE: The DemPref codebase creates a sampler via PyMC3 version 3.5;
            this codebase adapts their model to PyMC version 4.0.03b.

            We use the NUTS sampling algorithm (an extension of
            Hamilitonian Monte Carlo MCMC): https://arxiv.org/abs/1111.4246.
            """
            # Define update function:
            if self.update_func == "approx":

                def update_function(distribution):
                    result = at.sum(
                        [
                            -at.nnet.relu(
                                -self.beta_pref
                                * at.dot(self.phi_prefs[i], distribution)
                            )
                            for i in range(len(self.phi_prefs))
                        ]
                    ) + at.sum(
                        self.beta_demo * at.dot(self.phi_demos, distribution)
                    )
                    return result

            elif self.update_func == "pick_best":

                def update_function(distribution):
                    result = at.sum(
                        [
                            -at.log(
                                at.sum(
                                    at.exp(
                                        self.beta_pref
                                        * at.dot(
                                            self.phi_prefs[i], distribution
                                        )
                                    )
                                )
                            )
                            for i in range(len(self.phi_prefs))
                        ]
                    ) + at.sum(
                        self.beta_demo * at.dot(self.phi_demos, distribution)
                    )
                    return result

            elif self.update_func == "rank":

                def update_function(distribution):
                    result = (
                        at.sum(  # sum across different queries
                            [
                                at.sum(  # sum across different terms in PL-update
                                    -at.log(
                                        [
                                            at.sum(  # sum down different feature-differences in a single term in PL-update
                                                at.exp(
                                                    self.beta_pref
                                                    * at.dot(
                                                        self.phi_prefs[i][
                                                            j:, :
                                                        ]
                                                        - self.phi_prefs[i][j],
                                                        distribution,
                                                    )
                                                )
                                            )
                                            for j in range(
                                                self.query_option_count
                                            )
                                        ]
                                    )
                                )
                                for i in range(len(self.phi_prefs))
                            ]
                        )
                        + at.sum(
                            self.beta_demo
                            * at.dot(self.phi_demos, distribution)
                        ),
                    )
                    return result

            self.update_function = update_function

            while True:
                test_value = np.random.uniform(
                    low=-1, high=1, size=self.dim_features
                )
                test_value = test_value / np.linalg.norm(test_value)
                norm = (test_value ** 2).sum()
                if norm <= 1:
                    break

            # Get a sampling trace (and avoid Bad Initial Energy):
            while True:
                trace = self.get_trace(N, test_value)
                if trace is not None:
                    break
            if self._visualize:
                az.plot_trace(trace)
                plt.show()
                input("Press enter to continue")
                az.plot_energy(trace)
                plt.show()
                input("Press enter to continue")
                az.plot_posterior(trace)
                plt.show()
                input("Press enter to continue")
            all_samples = trace.sel(
                draw=slice(burn, None)
            ).posterior.rv_x.values
            all_samples = all_samples.reshape(
                all_samples.shape[0] * all_samples.shape[1], -1
            )
            w_samples = np.array([r / np.linalg.norm(r) for r in all_samples])

            return w_samples

        def get_trace(self, N: int, init_val: np.ndarray) -> az.InferenceData:
            """Create an MCMC trace."""
            # model accumulates the objects defined within the proceeding
            # context:
            model = pm.Model()
            with model:
                # Add random-variable x to model:
                rv_x = pm.Uniform(
                    name="rv_x",
                    shape=self.dim_features,
                    lower=-1,
                    upper=1,
                    initval=init_val,
                )

                # Define the prior as the unit ball centered at 0:
                def sphere(w):
                    """Determine if w is part of the unit ball.

                    NOTE: Original DemPref paper's sphere modeled
                    likelihood of out-of-distribution points as -np.inf;
                    doing so in PyMC now results in errors. -100, however,
                    represents log(1 / 1e45), which is a very small number.
                    """
                    w_sum = pm.math.sqr(w).sum()
                    result = at.switch(
                        pm.math.gt(w_sum, 1.0), -100, self.update_function(w)
                    )
                    return result

                try:
                    # Potential is a "potential term" defined as an "additional
                    # tensor...to be added to the model logp"(PyMC3 developer
                    # guide). In this instance, the potential is effectively
                    # the model's log-likelihood.
                    p = pm.Potential("sphere", sphere(rv_x))
                    trace = pm.sample(
                        N,
                        tune=1000,
                        return_inferencedata=True,
                        init="adapt_diag",
                        progressbar=False,
                        chains=4,
                    )
                except (
                    pm.SamplingError,
                    pm.parallel_sampling.ParallelSamplingError,
                ):
                    return None
            return trace

    class DemPrefQueryGenerator:
        """Generate queries.

        Code adapted from original DemPref agent.
        """

        def __init__(
            self,
            dom: Environment,
            num_queries: int,
            num_expectation_samples: int,
            include_previous_query: bool,
            generate_scenario: bool,
            update_func: str,
            beta_pref: float,
        ) -> None:
            """
            Initialize the approx query generation.

            Note: this class generates queries using approx gradients.

            ::original inputs:
                :dom: the domain to generate queries on
                :num_queries: number of queries to generate at each time step
                :trajectory_length: the length of each query
                :num_expectation_samples: number of w_samples to use in
                                          approximating the objective
                                          function
                :include_previous_query: boolean for whether one of the
                                         queries is the previously selected
                                         query
                :generate_scenario: boolean for whether we want to generate
                                    the scenario -- i.e., other agents'
                                    behavior
                :update_func: the update_func used; the options are
                              "pick_best", "approx", and "rank"
                :beta_pref: the rationality parameter for the teacher
                                  selecting her query
            ::Inquire-specific inputs:
                :start_state: The state from which a trajectory begins.
            """
            assert (
                num_queries >= 1
            ), "QueryGenerator.__init__: num_queries must be at least 1"
            assert (
                dom.trajectory_length >= 1
            ), "QueryGenerator.__init__: trajectory_length must be at least 1"
            assert (
                num_expectation_samples >= 1
            ), "QueryGenerator.__init__: num_expectation_samples must be \
                    at least 1"
            self.domain = dom
            self.num_queries = num_queries
            self.trajectory_length = dom.trajectory_length
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

        def generate_query_options(
            self,
            w_samples: np.ndarray,
            start_state: int,
            last_query_choice: Trajectory = None,
            blank_traj: bool = False,
        ) -> List[Trajectory]:
            """
            Generate self.num_queries number of query options.

            This function produces query options that (locally) maximize the
            maximum volume removal objective.

            :param w_samples: Samples of w
            :param last_query_choice: The previously selected query. Only
                                         required if self.incl_prev_query is
                                         True
            :param blank_traj: True is last_query_choice is blank. (Only
                               True if not using Dempref but using incl_prev_)
            :return: a list of trajectories (queries)
            """
            start = time.perf_counter()

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
                domain = args[0]
                w_samples = args[1]
                start_state = args[2]
                controls = np.array(controls)
                controls_set = [
                    controls[i * z : (i + 1) * z]
                    for i in range(self.num_new_queries)
                ]
                features_each_q_option = np.zeros(
                    (domain.w_dim(), self.num_new_queries)
                )
                for i, c in enumerate(controls_set):
                    features_each_q_option[:, i] = domain.trajectory_rollout(
                        start_state=start_state, actions=c
                    ).phi
                if self.include_previous_query and not blank_traj:
                    features_each_q_option = np.append(
                        features_each_q_option, last_query_choice.phi, axis=1
                    )
                if self.update_func == "pick_best":
                    return objective(features_each_q_option, w_samples)
                elif self.update_func == "approx":
                    return approx_objective(features_each_q_option, w_samples)
                else:
                    return rank_objective(features_each_q_option, w_samples)

            def objective(features: List, w_samples: np.ndarray) -> float:
                """
                Maximize the volume removal objective.

                :param features: a list containing the feature values of each
                                 query
                :param w_samples: samples of w, used to approximate the
                                  objective
                :return: the value of the objective function, evaluated on the
                         given queries' features
                """
                volumes_removed = []
                for i in range(len(features)):
                    feature_diff = np.array(
                        [f - features[i] for f in features]
                    )  # query_option_count x feature_size
                    weighted_feature_diff = (
                        np.sum(np.dot(feature_diff, w_samples.T), axis=1)
                        / w_samples.shape[0]
                    )  # query_option_count x 1 -- summed across w_samples
                    v_removed = 1.0 - 1.0 / np.sum(
                        np.exp(self.beta_pref * weighted_feature_diff)
                    )
                    volumes_removed.append(v_removed)
                return np.min(volumes_removed)

            def approx_objective(
                features: np.ndarray, w_samples: np.ndarray
            ) -> float:
                """
                Approximate the maximum volume removal objective.

                :param features: the feature values of each query option
                :param w_samples: w_samples of w used to approximate the
                                  objective
                :return: the value of the objective function, evaluated on the
                         given queries' features
                """
                if features.shape[0] > features.shape[1]:
                    features = features.T
                volumes_removed = []
                for i in range(len(features)):
                    feature_diff = (
                        features[i] - features[1 - i]
                    )  # 1 x feature_size
                    weighted_feature_diff = (
                        np.sum(np.dot(feature_diff, w_samples.T))
                        / w_samples.shape[0]
                    )  # 1 x 1 -- summed across w_samples
                    v_removed = 1.0 - np.minimum(
                        1.0, np.exp(self.beta_pref * weighted_feature_diff)
                    )
                    volumes_removed.append(v_removed)
                return np.min(volumes_removed)

            def rank_objective(features, w_samples) -> float:
                """
                The ranking maximum volume removal objective function.

                Note: This objective uses the Plackett-Luce model of
                teacher behavior.

                CANNOT BE USED WITH (incl_prev_QUERY AND NO DEMPREF).

                :param features: a list containing the feature values of each
                                 query
                :param w_samples: samples of w, used to approximate the
                                  objective
                :return: the value of the objective function, evaluated on the
                         given queries' features
                """
                # features: query_option_count x feature_size
                # w_samples: n_samples x feature_size
                exp_rewards = (
                    np.sum(np.dot(features, w_samples.T), axis=1)
                    / w_samples.shape[0]
                )  # query_option_count x 1 -- summed across w_samples
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

            action_space = self.domain.action_space()
            z = self.trajectory_length * action_space.dim
            lower_input_bound = list(action_space.min) * self.trajectory_length
            upper_input_bound = list(action_space.max) * self.trajectory_length
            u_sample = np.random.uniform(
                low=self.num_new_queries * lower_input_bound,
                high=self.num_new_queries * upper_input_bound,
                size=(self.num_new_queries * z),
            )
            query_options_controls = [
                u_sample[i * z : (i + 1) * z]
                for i in range(self.num_new_queries)
            ]
            end = time.perf_counter()
            print(f"Finished computing queries in {end - start}s")
            query_options_trajectories = [
                self.domain.trajectory_rollout(start_state, c)
                for c in query_options_controls
            ]
            if self.include_previous_query and not blank_traj:
                return [last_query_choice] + query_options_trajectories
            else:
                return query_options_trajectories
