import abc
from typing import Optional, Union
import pandas as pd


from autoencodix.utils._result import Result 
from autoencodix.utils._utils import nested_dict
from autoencodix.configs.default_config import DefaultConfig


class BaseVisualizer(abc.ABC):
    """ Defines the interface for visualizing training results.
    
    Attributes:
        plots: A nested dictionary to store various plots.
    """
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
        plot_type: str = "2D-scatter",
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[list, str]] = None,
        epoch: Optional[Union[int, None]] = None,
        split: str = "all",
    ) -> None:
        pass

    @abc.abstractmethod
    def show_weights(self) -> None:
        pass
