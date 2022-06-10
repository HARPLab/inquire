from abc import ABC, abstractmethod
from inquire.utils.datatypes import Query, Choice
from inquire.environments.environment import Task, CachedTask
from typing import Union

class Teacher(ABC):
    @abstractmethod
    def query_response(self, q: Query, task: Union[Task, CachedTask], verbose: bool) -> Choice:
        pass

    @property
    @abstractmethod
    def alpha(self):
        # alpha parameter for binary feedback
        pass
