

from dataclasses import field
from typing import Any, Dict
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict
from autoencodix.utils.default_config import DefaultConfig


class XModalVisualizer(BaseVisualizer):
    plots: Dict[str, Any] = field(
        default_factory=nested_dict
    )  ## Nested dictionary of plots as figure handles

    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        ## Make Model Weights plot
        self.plots["ModelWeights"] = self.plot_model_weights(model=result.model)

        ## Make long format of losses
        loss_df_melt = self.make_loss_format(result=result, config=config)

        ## Make plot loss absolute
        self.plots["loss_absolute"] = self.make_loss_plot(
            df_plot=loss_df_melt, plot_type="absolute"
        )
        ## Make plot loss relative
        self.plots["loss_relative"] = self.make_loss_plot(
            df_plot=loss_df_melt, plot_type="relative"
        )

        return result