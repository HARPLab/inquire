from inquire.teachers.teacher import Teacher
from inquire.utils.datatypes import Trajectory, Choice, Query, Feedback, Modality
from typing import Union
from inquire.utils.sampling import TrajectorySampling
from inquire.environments.environment import CachedTask, Task
import inquire.utils.learning
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from iam_domain_handler.domain_client import DomainClient



class HumanTeacher(Teacher):
    @property
    def alpha(self):
        return self._alpha

    def __init__(self) -> None:
        super().__init__()
        self._domain_client = DomainClient()
        self._alpha = 0.75

    def query_response(self, q: Query, task: Union[Task, CachedTask], verbose: bool=False) -> Choice:
        if q.query_type is Modality.DEMONSTRATION:
            f = self.demonstration(q, task)
            return f
        elif q.query_type is Modality.PREFERENCE:
            f = self.preference(q, task)
            return f
        elif q.query_type is Modality.CORRECTION:
            f = self.correction(q, task)
            return f
        elif q.query_type is Modality.BINARY:
            f = self.binary_feedback(q, task)
            return f
        else:
            raise Exception(self._type.__name__ + " does not support queries of type " + str(q.query_type))

    def demonstration(self, query: Query, task: Union[Task, CachedTask]) -> Choice:

        query_params = {
                    'instruction_text' : 'Please tap where the robot should place the next pepperoni.',
                    'display_type' : 3,
                    'bokeh_display_type' : 3,
                    'pizza' : {
                        'query_type' : 2,
                        'pizza_diameter' : task.domain.pizza_form['diameter'],
                        'crust_thickness' : task.domain.pizza_form['crust_thickness'],
                        'topping_diameter' : task.domain.pizza_form['topping_diam'],
                        'pizza_topping_positions_1_x' : query.trajectories[0].states[0].tolist(),
                        'pizza_topping_positions_1_y' : query.trajectories[0].states[1].tolist()
                    }
                }

        query_response = self._domain_client.run_query_until_done('Demonstration Query', query_params)
        new_point = query_response['query_point']

        traj = Trajectory(states=[np.append(query.trajectories[0].states[0], new_point[0]), 
                                  np.append(query.trajectories[0].states[1], new_point[1])], 
                          phi=query.trajectories[0].phi, actions=query.trajectories[0].actions) 
        return Feedback(Modality.DEMONSTRATION, query, Choice(traj, [traj] + query.trajectories))

    def preference(self, query: Query, task: Union[Task, CachedTask]) -> Choice:

        query_params = {
                    'instruction_text' : 'Please select which pizza is better where the next pepperoni placement is shown in green.',
                    'display_type' : 3,
                    'bokeh_display_type' : 3,
                    'pizza' : {
                        'query_type' : 0,
                        'pizza_diameter' : task.domain.pizza_form['diameter'],
                        'crust_thickness' : task.domain.pizza_form['crust_thickness'],
                        'topping_diameter' : task.domain.pizza_form['topping_diam'],
                        'pizza_topping_positions_1_x' : query.trajectories[0].states[0].tolist(),
                        'pizza_topping_positions_1_y' : query.trajectories[0].states[1].tolist(),
                        'pizza_topping_positions_2_x' : query.trajectories[1].states[0].tolist(),
                        'pizza_topping_positions_2_y' : query.trajectories[1].states[1].tolist()
                    }
                }
        query_response = self._domain_client.run_query_until_done('Preference Query', query_params)
        selected_pizza = query_response['query_response']

        return Feedback(Modality.PREFERENCE, query, Choice(selection=query.trajectories[selected_pizza], options=query.trajectories))

    def correction(self, query: Query, task: Union[Task, CachedTask]) -> Choice:
        
        query_params = {
                    'instruction_text' : 'Please drag and move the pepperoni to where the robot should place the next pepperoni.',
                    'display_type' : 3,
                    'bokeh_display_type' : 3,
                    'pizza' : {
                        'query_type' : 3,
                        'pizza_diameter' : task.domain.pizza_form['diameter'],
                        'crust_thickness' : task.domain.pizza_form['crust_thickness'],
                        'topping_diameter' : task.domain.pizza_form['topping_diam'],
                        'pizza_topping_positions_1_x' : query.trajectories[0].states[0].tolist(),
                        'pizza_topping_positions_1_y' : query.trajectories[0].states[1].tolist()
                    }
                }

        query_response = self._domain_client.run_query_until_done('Correction Query', query_params)
        new_point = query_response['query_point']

        correction = query.trajectories[0]
        correction.states[0][-1] = new_point[0]
        correction.states[1][-1] = new_point[1]

        return Feedback(Modality.CORRECTION, query, Choice(selection=correction, options=[correction, query.trajectories[0]]))

    def binary_feedback(self, query: Query, task: Union[Task, CachedTask], verbose: bool=False) -> Choice:
        assert(len(query.trajectories) == 1)

        query_params = {
                    'instruction_text' : 'Please decide whether the next pepperoni placement that is indicated in green is good or bad.',
                    'display_type' : 3,
                    'bokeh_display_type' : 3,
                    'pizza' : {
                        'query_type' : 1,
                        'pizza_diameter' : task.domain.pizza_form['diameter'],
                        'crust_thickness' : task.domain.pizza_form['crust_thickness'],
                        'topping_diameter' : task.domain.pizza_form['topping_diam'],
                        'pizza_topping_positions_1_x' : query.trajectories[0].states[0].tolist(),
                        'pizza_topping_positions_1_y' : query.trajectories[0].states[1].tolist()
                    }
                }

        query_response = self._domain_client.run_query_until_done('Binary Query', query_params)
        bin_fb = query_response['query_response'] 

        return Feedback(query.query_type, query, Choice(bin_fb, [query.trajectories[0]]))
