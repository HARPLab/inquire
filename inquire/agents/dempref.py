"""
A demonstrations and preferences agent.

Adapted from Learning Reward Functions
by Integrating Human Demonstrations and Preferences.
"""
from agent import Agent


class DemPref(Agent):
    """A preference-querying agent seeded with demonstrations."""

    def __init__(
            self,
            sampling_method,
            optional_sampling_params,
            weight_sample_count: int,
            trajectory_sample_count: int,
            trajectory_length: int,
            interaction_types: list = []
            ):
        """Initialize the agent."""
        pass

    def reset(self):
        """Prepare for new query session."""
        pass

    def generate_query(
            self,
            domain,
            query_state,
            curr_w,
            verbose: bool = False
            ) -> list:
        """Generate query using approximate gradients.

        Code adapted from DemPref's ApproxQueryGenerator.
        """
        pass

    def update_weights(self, domain, feedback):
        """Update the model's learned weights."""
        pass

    def approx_volume_removal(self) -> None:
        """Volume removal objective function."""
        pass
