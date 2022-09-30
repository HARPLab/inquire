import argparse
import math
import os
import pdb
import time

import numpy as np
from inquire.utils.datatypes import Modality
from inquire.utils.sampling import TrajectorySampling


class ArgsHandler:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Parameters for evaluating INQUIRE"
        )
        parser.add_argument(
            "-V",
            "--verbose",
            dest="verbose",
            action="store_true",
            help="verbose",
        )
        parser.add_argument(
            "--use_cache",
            dest="use_cache",
            action="store_true",
            help="use cached trajectories instead of sampling",
        )
        parser.add_argument(
            "--numba",
            dest="use_numba",
            action="store_true",
            help="use cached trajectories instead of sampling",
        )
        parser.add_argument(
            "--reuse_weights",
            dest="reuse_weights",
            action="store_true",
            help="reuse the weight distribution resulting from the previous query",
        )
        parser.add_argument(
            "--static_state",
            dest="static_state",
            action="store_true",
            help="use the same state for all queries",
        )
        parser.add_argument(
            "--teacher_displays",
            action="store_true",
            help="display the teacher's interactions.",
        )
        parser.add_argument(
            "-K",
            "--queries",
            type=int,
            dest="num_queries",
            default=5,
            help="number of queries",
        )
        parser.add_argument(
            "-R",
            "--runs",
            type=int,
            dest="num_runs",
            default=10,
            help="number of evaluations to run for each task",
        )
        parser.add_argument(
            "-X",
            "--tests",
            type=int,
            dest="num_test_states",
            default=1,
            help="number of test states to evaluate",
        )
        parser.add_argument(
            "-Z",
            "--tasks",
            type=int,
            dest="num_tasks",
            default=1,
            help="number of task instances to generate",
        )
        parser.add_argument(
            "--beta",
            type=float,
            dest="beta",
            default=1.0,
            help="optimality parameter",
        )
        parser.add_argument(
            "--betas",
            type=str,
            dest="beta_vals",
            help="optimality parameter, set for each interaction type",
        )
        parser.add_argument(
            "--costs",
            type=str,
            dest="cost_vals",
            help="cost per interaction, set for each interaction type",
        )
        parser.add_argument(
            "-C",
            "--convergence_threshold",
            type=float,
            dest="conv_threshold",
            default=1.0e-1,
            help="convergence threshold for estimating weight distribution",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            dest="step_size",
            default=0.05,
            help="step size for weight optimization",
        )
        parser.add_argument(
            "-M",
            type=int,
            dest="num_w_samples",
            default=100,
            help="number of weight samples",
        )
        parser.add_argument(
            "-N",
            type=int,
            dest="num_traj_samples",
            default=50,
            help="number of trajectory samples",
        )
        parser.add_argument(
            "-D",
            "--domain",
            type=str,
            dest="domain_name",
            default="linear_combo",
            choices=[
                "lander",
                "linear_combo",
                "linear_system",
                "pats_linear_system",
                "pizza",
            ],
            help="name of the evaluation domain",
        )
        parser.add_argument(
            "-I",
            "--opt_iterations",
            type=int,
            dest="opt_iters",
            default=50,
            help="number of attempts to optimize a sample of controls (pertinent to lunar lander, linear system, and pizza-making domains)",
        )
        parser.add_argument(
            "-S",
            "--sampling",
            type=str,
            dest="sampling_method",
            default="uniform",
            choices=["uniform"],
            help="name of the trajectory sampling method",
        )
        parser.add_argument(
            "-A",
            "--agent",
            type=str,
            dest="agent_name",
            default="inquire",
            choices=[
                "inquire",
                "dempref",
                "biased_dempref",
                "no-demos",
                "demo-only",
                "pref-only",
                "corr-only",
                "binary-only",
                "all",
                "titrated",
            ],
            help="name of the agent to evaluate",
        )
        parser.add_argument(
            "-T",
            "--teacher",
            type=str,
            dest="teacher_name",
            default="optimal",
            choices=["optimal"],
            help="name of the simulated teacher to query",
        )
        parser.add_argument(
            "-O",
            "--output",
            type=str,
            dest="output_dir",
            default="output",
            help="name of the output directory",
        )
        parser.add_argument(
            "--output_name",
            type=str,
            dest="output_name",
            help="name of the output filename",
        )
        parser.add_argument(
            "--seed_with_n_demos",
            type=int,
            dest="n_demos",
            default=1,
            help="how many demos to provide before commencing preference queries. Specific to DemPref.",
        )

        self._args = parser.parse_args()

        ## Define externally available variables
        self.num_tasks = self._args.num_tasks
        self.num_runs = self._args.num_runs
        self.num_queries = self._args.num_queries
        self.num_test_states = self._args.num_test_states
        self.num_traj_samples = self._args.num_traj_samples
        self.step_size = self._args.step_size
        self.conv_threshold = self._args.conv_threshold
        self.use_cache = self._args.use_cache
        self.static_state = self._args.static_state
        self.reuse_weights = self._args.reuse_weights
        self.verbose = self._args.verbose
        self.output_name = self._args.output_name
        self.output_dir = self._args.output_dir

    def setup_domain(self):
        ## Set up domain
        if self._args.domain_name == "linear_combo":
            from inquire.environments.linear_combo import LinearCombination

            seed = 42
            w_dim = 8
            domain = LinearCombination(seed, w_dim)

        elif self._args.domain_name == "lander":
            from inquire.environments.lunar_lander import LunarLander

            traj_length = 10
            optimization_iteration_count = self._args.opt_iters
            if self._args.agent_name == "biased_dempref":
                domain = LunarLander(
                    optimal_trajectory_iterations=optimization_iteration_count,
                    verbose=self._args.verbose,
                    include_feature_biases=True,
                )
            else:
                domain = LunarLander(
                    optimal_trajectory_iterations=optimization_iteration_count,
                    verbose=self._args.verbose,
                )

        elif self._args.domain_name == "pats_linear_system":
            from inquire.environments.pats_linear_dynamical_system import \
                PatsLinearDynamicalSystem

            traj_length = 15
            optimization_iteration_count = self._args.opt_iters
            domain = PatsLinearDynamicalSystem(
                trajectory_length=traj_length,
                optimal_trajectory_iterations=optimization_iteration_count,
                verbose=self._args.verbose,
            )

        elif self._args.domain_name == "linear_system":
            from inquire.environments.linear_dynamical_system import \
                LinearDynamicalSystem

            traj_length = 15
            optimization_iteration_count = self._args.opt_iters
            domain = LinearDynamicalSystem(
                trajectory_length=traj_length,
                optimal_trajectory_iterations=optimization_iteration_count,
                verbose=self._args.verbose,
            )
        elif self._args.domain_name == "pizza":
            from inquire.environments.pizza_making import PizzaMaking

            traj_length = how_many_toppings_to_add = 1
            max_topping_count = 15
            pizza_form = {
                "diameter": 35,
                "crust_thickness": 2.54,
                "topping_diam": 3.54,
            }
            basis_functions = [
                "x_coordinate",
                "y_coordinate",
                "dist_0_quadratic",
                "dist_4_quadratic",
            ]
            domain = PizzaMaking(
                max_topping_count=max_topping_count,
                how_many_toppings_to_add=how_many_toppings_to_add,
                pizza_form=pizza_form,
                basis_functions=basis_functions,
                verbose=self._args.verbose,
            )
        self.w_dim = domain.w_dim()
        return domain

    def setup_agents(self):
        ## Set up sampling method
        if self._args.sampling_method == "uniform":
            sampling_method = TrajectorySampling.uniform_sampling
            sampling_params = {}
        else:
            raise ValueError("Unknown trajectory sampling method")

        ## Set up agent(s)
        if self._args.beta_vals is None:
            beta = self._args.beta
        else:
            beta = eval(self._args.beta_vals)

        if self._args.agent_name == "titrated":
            from inquire.agents.inquire import FixedInteractions

            ddddd = FixedInteractions(
                sampling_method,
                sampling_params,
                self._args.num_w_samples,
                self._args.num_traj_samples,
                [Modality.DEMONSTRATION] * 5,
                beta=beta
            )
            ddddp = FixedInteractions(
                sampling_method,
                sampling_params,
                self._args.num_w_samples,
                self._args.num_traj_samples,
                [Modality.DEMONSTRATION] * 4 + [Modality.PREFERENCE],
                beta=beta
            )
            dddpp = FixedInteractions(
                sampling_method,
                sampling_params,
                self._args.num_w_samples,
                self._args.num_traj_samples,
                [Modality.DEMONSTRATION] * 3 + [Modality.PREFERENCE] * 2,
                beta=beta
            )
            ddppp = FixedInteractions(
                sampling_method,
                sampling_params,
                self._args.num_w_samples,
                self._args.num_traj_samples,
                [Modality.DEMONSTRATION] * 2 + [Modality.PREFERENCE] * 3,
                beta=beta
            )
            dpppp = FixedInteractions(
                sampling_method,
                sampling_params,
                self._args.num_w_samples,
                self._args.num_traj_samples,
                [Modality.DEMONSTRATION] + [Modality.PREFERENCE] * 4,
                beta=beta
            )
            ppppp = FixedInteractions(
                sampling_method,
                sampling_params,
                self._args.num_w_samples,
                self._args.num_traj_samples,
                [Modality.PREFERENCE] * 5,
                beta=beta
            )
            agents = [ddddd, ddddp, dddpp, ddppp, dpppp, ppppp]
            agent_names = [
                "DDDDD",
                "DDDDP",
                "DDDPP",
                "DDPPP",
                "DPPPP",
                "PPPPP",
            ]
        if (
            self._args.agent_name.lower() == "dempref"
            or self._args.agent_name.lower() == "biased_dempref"
        ):
            from inquire.agents.dempref import DemPref

            agents = [
                DemPref(
                    weight_sample_count=self._args.num_w_samples,
                    trajectory_sample_count=self._args.num_traj_samples,
                    interaction_types=[
                        Modality.DEMONSTRATION,
                        Modality.PREFERENCE,
                    ],
                    w_dim=self.w_dim,
                    seed_with_n_demos=self._args.n_demos,
                    domain_name=self._args.domain_name,
                )
            ]
            agent_names = ["DEMPREF"]
        if self._args.cost_vals is None:
            costs = None
        else:
            costs = eval(self._args.cost_vals)
        use_numba = self._args.use_numba
        if self._args.agent_name == "inquire":
            from inquire.agents.inquire import Inquire

            agents = [
                Inquire(
                    sampling_method,
                    sampling_params,
                    self._args.num_w_samples,
                    self._args.num_traj_samples,
                    [
                        Modality.DEMONSTRATION,
                        Modality.PREFERENCE,
                        Modality.CORRECTION,
                        Modality.BINARY,
                    ],
                    beta=beta,
                    costs=costs,
                    use_numba=use_numba,
                )
            ]
            agent_names = ["INQUIRE"]
        elif self._args.agent_name == "no-demos":
            from inquire.agents.inquire import Inquire

            agents = [
                Inquire(
                    sampling_method,
                    sampling_params,
                    self._args.num_w_samples,
                    self._args.num_traj_samples,
                    [
                        Modality.PREFERENCE,
                        Modality.CORRECTION,
                        Modality.BINARY,
                    ],
                    beta=beta,
                    costs=costs,
                    use_numba=use_numba,
                )
            ]
            agent_names = ["INQUIRE wo/Demos"]
        elif self._args.agent_name == "demo-only":
            from inquire.agents.inquire import Inquire

            agents = [
                Inquire(
                    sampling_method,
                    sampling_params,
                    self._args.num_w_samples,
                    self._args.num_traj_samples,
                    [Modality.DEMONSTRATION],
                    beta=beta,
                    use_numba=use_numba,
                )
            ]
            agent_names = ["Demo-only"]
        elif self._args.agent_name == "pref-only":
            from inquire.agents.inquire import Inquire

            agents = [
                Inquire(
                    sampling_method,
                    sampling_params,
                    self._args.num_w_samples,
                    self._args.num_traj_samples,
                    [Modality.PREFERENCE],
                    beta=beta,
                    use_numba=use_numba,
                )
            ]
            agent_names = ["Pref-only"]
        elif self._args.agent_name == "corr-only":
            from inquire.agents.inquire import Inquire

            agents = [
                Inquire(
                    sampling_method,
                    sampling_params,
                    self._args.num_w_samples,
                    self._args.num_traj_samples,
                    [Modality.CORRECTION],
                    beta=beta,
                    use_numba=use_numba,
                )
            ]
            agent_names = ["Corr-only"]
        elif self._args.agent_name == "binary-only":
            from inquire.agents.inquire import Inquire

            agents = [
                Inquire(
                    sampling_method,
                    sampling_params,
                    self._args.num_w_samples,
                    self._args.num_traj_samples,
                    [Modality.BINARY],
                    beta=beta,
                    use_numba=use_numba,
                )
            ]
            agent_names = ["Binary-only"]
        return agents, agent_names

    def setup_teacher(self):
        ## Set up teacher
        if self._args.teacher_name == "optimal":
            from inquire.teachers.optimal import OptimalTeacher

            teacher = OptimalTeacher(
                self._args.num_traj_samples, self._args.teacher_displays
            )
        return teacher
