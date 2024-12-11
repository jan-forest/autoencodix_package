import abc
from src.autoencodix.utils._result import Result


class BaseTrainer(abc.ABC):
    """
    Interface for Trainer classes, custom trainers should follow this interface

    Attributes
    ----------
    _result : Result
        Result object to store the training results

    Methods
    -------
    train():
        Abstract method to train the model


    """

    def __init__(self):
        self._result = Result()
        pass

    @abc.abstractmethod
    def train(self):
        pass
