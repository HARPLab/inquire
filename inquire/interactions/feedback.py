class Query:
    def __init__(self, query_type, task, start_state, trajectories):
        self.query_type = query_type
        self.task = task  # A task has a domain as an instance attribute
        self.start_state = start_state
        self.trajectories = trajectories


class Trajectory:
    def __init__(self, trajectory, phi):
        self.trajectory = trajectory
        self.phi = phi


class Choice:
    def __init__(self, selection, options):
        self.selection = selection
        self.options = options
