"""
An agent which uses demonstrations and preferences.

Code adapted from Learning Reward Functions
by Integrating Human Demonstrations and Preferences.
"""
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from inquire.agents.agent import Agent
from inquire.environments.environment import Environment
from inquire.interactions.feedback import Query, Trajectory
from inquire.interactions.modalities import Preference

import numpy as np

import pandas as pd

import pymc3 as pm

import scipy.optimize as opt

import theano as th
import theano.tensor as tt


class DemPref(Agent):
    """A preference-querying agent seeded with demonstrations.

    Note: We instantiate the agent according to arguments corresponding to
    what the the original paper's codebase designates as their main experiment.
    """

    def __init__(
        self,
        weight_sample_count: int,
        trajectory_sample_count: int,
        trajectory_length: int,
        interaction_types: list = [],
        w_dim: int = 4,
        which_param_csv: int = 0,
    ):
        """Initialize the agent.

        Note we needn't maintain a domain's start state; that's handled in
        inquire/tests/evaluation.py and the respective domain.
        """
        self._weight_sample_count = weight_sample_count
        self._trajectory_sample_count = trajectory_sample_count
        self._trajectory_length = trajectory_length
        self._interaction_types = interaction_types

        """
        Get the pre-defined agent parameters
        """
        self._dempref_agent_parameters = self.read_param_csv(which_param_csv)

        """
        Instance attributes from orginal codebase's 'runner.py' object. Note
        that some variable names are modified to be consist with the Inquire
        parlance.
        """
        self.domain_name = self._dempref_agent_parameters["domain"][0]
        self.teacher_type = self._dempref_agent_parameters["teacher_type"][0]

        self.n_demos = self._dempref_agent_parameters["n_demos"][0]
        self.gen_demos = self._dempref_agent_parameters["gen_demos"][0]
        self.opt_iter_count = self._dempref_agent_parameters["opt_iter_count"][
            0
        ]
        self.trim_start = self._dempref_agent_parameters["trim_start"][0]

        self.query_option_count = self._dempref_agent_parameters[
            "query_option_count"
        ][0]
        self.update_func = self._dempref_agent_parameters["update_func"][0]
        self.trajectory_length = self._dempref_agent_parameters[
            "trajectory_length"
        ][0]
        self.incl_prev_query = self._dempref_agent_parameters[
            "incl_prev_query"
        ][0]
        self.gen_scenario = self._dempref_agent_parameters["gen_scenario"][0]
        self.n_pref_iters = self._dempref_agent_parameters["n_pref_iters"][0]
        self.epsilon = self._dempref_agent_parameters["epsilon"][0]

        """
        Instantiate the DemPref-specific sampler and query generator:
        """
        self._sampler = None
        self._w_samples = None
        self._query_generator = None
        self._first_q_session = True
        self._q_session_index = 0
        self._query_index = 0
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

        self.n_samples_summ = self._dempref_agent_parameters["n_samples_summ"][
            0
        ]
        self.n_samples_exp = self._dempref_agent_parameters["n_samples_exp"][0]
        self.beta_demo = self._dempref_agent_parameters["beta_demo"][0]
        self.beta_pref = self._dempref_agent_parameters["beta_pref"][0]
        self.beta_teacher = self._dempref_agent_parameters["beta_teacher"][0]

        """If we want to save data as they did in DemPref:"""
        # self.config = [
        #    self.teacher_type,
        #    self.n_demos,
        #    self.trim_start,
        #    self.query_option_count,
        #    self.update_func,
        #    self.trajectory_length,
        #    self.incl_prev_query,
        #    self.gen_scenario,
        #    self.n_pref_iters,
        #    self.epsilon,
        #    self.n_samples_summ,
        #    self.n_samples_exp,
        #    self.beta_demo,
        #    self.beta_pref,
        #    self.beta_teacher,
        # ]
        # self.df = pd.DataFrame(columns=["run #", "pref_iter", "type", "value"])

    def reset(self) -> None:
        """Prepare for new query session."""
        self._sampler.clear_pref()
        self._sampler = self.DemPrefSampler(
            query_option_count=self.query_option_count,
            dim_features=self._w_dim,
            update_func=self.update_func,
            beta_demo=self.beta_demo,
            beta_pref=self.beta_pref,
        )
        self.w_samples = self._sampler.sample(N=self.n_samples_summ)
        """If we want to save data as they did in DemPref:"""
        # mean_w = np.mean(self.w_samples, axis=0)
        # mean_w = mean_w / np.linalg.norm(mean_w)
        # var_w = np.var(self.w_samples, axis=0)
        ## Make sure to properly index data:
        # if self.first_q_session:
        #    self.first_q_session = False
        # else:
        #    self.q_session_index += 1
        #    self.query_index = 0
        # data = [
        #    [self.q_session_index + 1, 0, "mean", mean_w],
        #    [self.q_session_index + 1, 0, "var", var_w],
        # ]
        # self.df = self.df.append(
        #    pd.DataFrame(
        #        data, columns=["run #", "pref_iter", "type", "value"]
        #    ),
        #    ignore_index=True,
        # )

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
                trajectory_length=self.trajectory_length,
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
                    query_options = self._query_generator.generate_query_options(
                        self.w_samples, blank_traj=True
                    )
                else:
                    query_options = self._query_generator.generate_query_options(
                        self.w_samples, last_query_choice
                    )
            else:
                query_options = self._query_generator.generate_query_options(
                    self.w_samples
                )
            query_diffs = []
            for m in range(len(query_options)):
                for n in range(m):
                    query_diffs.append(
                        np.linalg.norm(
                            domain.features_from_trajectory(
                                query_options[m].trajectory
                            )
                            - domain.features_from_trajectory(
                                query_options[n].trajectory
                            )
                        )
                    )
            query_diff = max(query_diffs)

        query = Query(
            query_type=Preference,
            task=None,
            start_state=query_state,
            trajectories=query_options,
        )

        return query

    def update_weights(
        self, domain: Environment, feedback: list
    ) -> np.ndarray:
        """Update the model's learned weights."""
        if feedback == []:
            # No feedback to consider at this point
            return
        else:
            # Use the most recent Choice in feedback:
            query_options = feedback[-1].options
            choice = feedback[-1].selection
            choice_index = query_options.index(choice)
            if self.incl_prev_query:
                self.all_query_choices[self.random_scenario_index] = choice

            # Create dictionary map from rankings to query-option features;
            # load into sampler:
            features = [
                domain.features_from_trajectory(x.trajectory)
                for x in query_options
            ]
            phi = {k: features[k] for k in range(len(query_options))}
            self._sampler.load_prefs(phi, choice_index)
            self.w_samples = self._sampler.sample(N=self.n_samples_summ)
            # Return the new weights from the samples:
            mean_w = np.mean(self.w_samples, axis=0)
            mean_w = mean_w / np.linalg.norm(mean_w)
            return np.array(mean_w, copy=True).reshape(1, -1)

    def read_param_csv(self, which_csv: int = 0) -> dict:
        """Read an agent-parameterization .csv.

        ::inputs:
            :creation_index: A time-descending .csv file index.
                      e.g. if creation_index = 0, use the dempref
                      dempref_agent.csv most recently created.
        """
        data_path = Path.cwd() / Path("../inquire/agents/")
        # Sort the .csvs in descending order by time of creation:
        all_files = np.array(list(Path.iterdir(data_path)))
        all_csvs = all_files[
            np.argwhere([f.suffix == ".csv" for f in all_files])
        ]
        all_csvs = np.array([str(f[0]).strip() for f in all_csvs])
        sorted_csvs = sorted(all_csvs, key=os.path.getmtime)
        sorted_csvs = [Path(c) for c in sorted_csvs]
        # Select the indicated .csv and convert it to a dictionary:
        chosen_csv = sorted_csvs[-which_csv]
        df = pd.read_csv(chosen_csv)
        params_dict = df.to_dict()
        return params_dict

    def process_demonstrations(
        self, trajectories: list, domain: Environment
    ) -> None:
        """Generate demonstrations to seed the querying process."""
        self.demos = trajectories
        phi_demos = [
            domain.features_from_trajectory(x.trajectory) for x in self.demos
        ]
        self._sampler.load_demo(np.array(phi_demos))
        self.cleaned_demos = self.demos
        if self.incl_prev_query:
            self.all_query_choices = [d for d in self.cleaned_demos]

    class DemPrefSampler:
        """Sample trajectories for querying.

        Code adapted from original DemPref agent.
        """

        def __init__(
            self,
            query_option_count: int,
            dim_features: int,
            update_func: str = "pick_best",
            beta_demo: float = 0.1,
            beta_pref: float = 1.0,
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

        def sample(self, N: int, T: int = 1, burn: int = 100000) -> np.ndarray:
            """Return N samples from the distribution.

            The distribution is defined by applying update_func on the
            demonstrations and preferences observed thus far.

            :param N: number of w_samples to draw.
            :param T: if greater than 1, all samples except each T^{th}
                      sample are discarded
            :param burn: how many samples before the chain converges;
                         these initial samples are discarded
            :return: list of w_samples drawn
            """
            x = tt.vector()
            x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)

            # Define update function:
            start = time.perf_counter()
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
                    tt.sum(  # sum across different queries
                        [
                            tt.sum(  # sum across different terms in PL-update
                                -tt.log(
                                    [
                                        tt.sum(  # sum down different feature-differences in a single term in PL-update
                                            tt.exp(
                                                self.beta_pref
                                                * tt.dot(
                                                    self.phi_prefs[i][j:, :]
                                                    - self.phi_prefs[i][j],
                                                    x,
                                                )
                                            )
                                        )
                                        for j in range(self.query_option_count)
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
                f"{time.perf_counter() - start}s"
            )

            """Define model for MCMC.

            NOTE the DemPref codebase creates a sampler via PyMC3 version 3.5;
            this codebase adapts their model to PyMC3 version 3.11.2.

            We use the NUTS sampling algorithm (an extension of
            Hamilitonian Monte Carlo MCMC): https://arxiv.org/abs/1111.4246.
            """

            # model accumulates the objects defined within the proceeding
            # context:
            with pm.Model() as model:
                # Add random-variable x to model:
                rv_x = pm.Uniform(
                    "x",
                    shape=(self.dim_features,),
                    lower=-1,
                    upper=1,
                    testval=np.zeros(self.dim_features),  # The initial values
                )

                # Define the log-likelihood function:
                def sphere(rv):
                    if tt.le(1.0, (x ** 2).sum()):
                        # DemPref used -np.inf which yields a 'bad initial
                        # energy' error. Use sys.maxsize instead:
                        return tt.as_tensor_variable(-sys.maxsize)
                    else:
                        return tt.as_tensor_variable(self.f(rv))

                # Potential is a "potential term" defined as an
                # "additional tensor...to be added to the model logp"
                # (PYMC3 developer guide):

                p = pm.Potential(name="sphere", var=sphere(rv_x))
                trace = pm.sample(
                    n_init=200000,
                    tune=burn,
                    discard_tuned_samples=True,
                    progressbar=True,
                    return_inferencedata=False,
                )
            all_samples = trace.get_values(varname=x)
            w_samples = np.array([r / np.linalg.norm(r) for r in all_samples])

            # print(f"Finished MCMC after drawing {N*T+burn)} w_samples")
            return w_samples

    class DemPrefQueryGenerator:
        """Generate queries.

        Code adapted from original DemPref agent.
        """

        def __init__(
            self,
            dom: Environment,
            num_queries: int,
            trajectory_length: int,
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
                trajectory_length >= 1
            ), "QueryGenerator.__init__: trajectory_length must be at least 1"
            assert (
                num_expectation_samples >= 1
            ), "QueryGenerator.__init__: num_expectation_samples must be \
                    at least 1"
            self.domain = dom
            self.num_queries = num_queries
            self.trajectory_length = trajectory_length
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
            last_query_choice: Trajectory = None,
            blank_traj: bool = False,
        ) -> List[Trajectory]:
            """
            Generate self.num_queries number of queries.

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
                controls = np.array(controls)
                controls_set = [
                    controls[i * z : (i + 1) * z]
                    for i in range(self.num_new_queries)
                ]
                features_each_q_option = np.zeros(
                    (domain.w_dim, self.num_new_queries)
                )
                for i, c in enumerate(controls_set):
                    features_each_q_option[
                        :, i
                    ] = domain.features_from_trajectory(
                        c, controls_as_input=True
                    )
                if self.include_previous_query and not blank_traj:
                    features_each_q_option = np.append(
                        features_each_q_option,
                        domain.features_from_trajectory(last_query_choice),
                        axis=1,
                    )
                if self.update_func == "pick_best":
                    return -objective(features_each_q_option, w_samples)
                elif self.update_func == "approx":
                    return -approx_objective(features_each_q_option, w_samples)
                else:
                    return -rank_objective(features_each_q_option, w_samples)

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

            # The following optimization is w.r.t. volume removal; the domain's
            # optimization is w.r.t. the linear combination of weights and
            # features; this difference is a trait of the DemPref codebase.
            z = self.trajectory_length * self.domain.control_size
            lower_input_bound = [
                x[0] for x in self.domain.control_bounds
            ] * self.trajectory_length
            upper_input_bound = [
                x[1] for x in self.domain.control_bounds
            ] * self.trajectory_length
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
                * self.trajectory_length,
                approx_grad=True,
            )
            query_options_controls = [
                opt_res[0][i * z : (i + 1) * z]
                for i in range(self.num_new_queries)
            ]
            end = time.perf_counter()
            print(f"Finished computing queries in {end - start}s")
            # Note the domain was reset w/ appropriate seed before beginning
            # this query session; domain.run(c) will thus reset to appropriate
            # state:
            raw_trajectories = [
                self.domain.run(c) for c in query_options_controls
            ]
            raw_phis = [
                self.domain.features_from_trajectory(t)
                for t in raw_trajectories
            ]
            query_options_trajectories = [
                Trajectory(raw_trajectories[i], raw_phis[i])
                for i in range(len(raw_trajectories))
            ]
            if self.include_previous_query and not blank_traj:
                return [last_query_choice] + query_options_trajectories
            else:
                return query_options_trajectories
