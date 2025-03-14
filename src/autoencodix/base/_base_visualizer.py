import abc
from typing import Any, Optional, Dict, Union


from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


class BaseVisualizer(abc.ABC):
    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

    @abc.abstractmethod
    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        pass

    @abc.abstractmethod
    def show_loss(self, type: str = "absolute") -> None:
        pass

    @abc.abstractmethod
    def save_plots(self, path: str, which: str = "all", format: str = "png") -> None:
        pass

    @abc.abstractmethod
    def show_latent_space(
        self,
        result: Result,
        type: str = "2D-scatter",
        label_list: Optional[Union[list, None]] = None,
        param: str = "all",
        epoch: Optional[Union[int, None]]  = None,
        split: str = "all",
    ) -> None:
        pass

    @abc.abstractmethod
    def show_weights(self) -> None:
        pass
