from abc import ABC, abstractmethod
from inquire.interactions.feedback import Query, Choice

class Teacher(ABC):
    @abstractmethod
    def query_response(self, q: Query, verbose: bool) -> Choice:
        pass

    @property
    @abstractmethod
    def alpha(self):
        # alpha parameter for binary feedback
        pass
