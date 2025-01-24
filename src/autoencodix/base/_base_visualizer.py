import abc
from autoencodix.utils._result import Result


class BaseVisualizer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def visualize(self, result: Result) -> Result:
        pass
