from abc import ABC, abstractmethod
from inquire.interactions.feedback import Query, Choice

class Teacher(ABC):
    @abstractmethod
    def query(self, q: Query, verbose: bool) -> Choice:
        pass

    # @property
    # @abstractmethod
    # def phi(self):
    #     pass
