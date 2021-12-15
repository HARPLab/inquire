from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_query(self, domain, query_state, curr_w):
        pass

    @abstractmethod
    def update_weights(self, domain, feedback):
        pass
