import abc
import os
from typing import Optional, Union
import pandas as pd
from matplotlib import pyplot as plt

from autoencodix.utils._result import Result 
from autoencodix.utils._utils import nested_dict, nested_to_tuple, show_figure
from autoencodix.utils.default_config import DefaultConfig


class BaseVisualizer(abc.ABC):
    """ Defines the interface for visualizing training results.
    
    Attributes:
        plots: A nested dictionary to store various plots.
    """
    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

### Abstract Methods ###
    @abc.abstractmethod
    def visualize(self, result: Result, config: DefaultConfig) -> Result:
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

### General Functions used by all Visualizers in similar way ###

   ## TODO could be moved to BaseVisualizer?
    def show_loss(self, plot_type: str = "absolute") -> None:
        """
        Display the loss plot.
        Parameters:
        plot_type (str): The type of loss plot to display.
                    Options are "absolute" for the absolute loss plot and
                    "relative" for the relative loss plot.
                    Defaults to "absolute".
        Returns:
        None
        """
        if plot_type == "absolute":
            if "loss_absolute" not in self.plots.keys():
                print("Absolute loss plot not found in the plots dictionary")
                print("You need to run visualize() method first")
            else:
                fig = self.plots["loss_absolute"]
                show_figure(fig)
                plt.show()
        if plot_type == "relative":
            if "loss_relative" not in self.plots.keys():
                print("Relative loss plot not found in the plots dictionary")
                print("You need to run visualize() method first")
            else:
                fig = self.plots["loss_relative"]
                show_figure(fig)
                plt.show()

        if plot_type not in ["absolute", "relative"]:
            print(
                "Type of loss plot not recognized. Please use 'absolute' or 'relative'"
            )

	## TODO move to BaseVisualizer?
    def save_plots(
        self, path: str, which: Union[str, list] = "all", format: str = "png"
    ) -> None:
        """
        Save specified plots to the given path in the specified format.

        Parameters:
        path (str): The directory path where the plots will be saved.
        which (list or str): A list of plot names to save or a string specifying which plots to save.
                             If 'all', all plots in the plots dictionary will be saved.
                             If a single plot name is provided as a string, only that plot will be saved.
        format (str): The file format in which to save the plots (e.g., 'png', 'jpg').

        Returns:
        None

        Raises:
        ValueError: If the 'which' parameter is not a list or a string.
        """
        if not isinstance(which, list):
            ## Case when which is a string
            if which == "all":
                ## Case when all plots are to be saved
                if len(self.plots) == 0:
                    print("No plots found in the plots dictionary")
                    print("You need to run  visualize() method first")
                else:
                    for item in nested_to_tuple(self.plots):
                        fig = item[-1]  ## Figure is in last element of the tuple
                        filename = "_".join(str(x) for x in item[0:-1])
                        fullpath = os.path.join(path, filename)
                        fig.savefig(f"{fullpath}.{format}")
            else:
                ## Case when a single plot is provided as string
                if which not in self.plots.keys():
                    print(f"Plot {which} not found in the plots dictionary")
                    print(f"All available plots are: {list(self.plots.keys())}")
                else:
                    for item in nested_to_tuple(
                        self.plots[which]
                    ):  # Plot all epochs and splits of type which
                        fig = item[-1]  ## Figure is in last element of the tuple
                        filename = which + "_" + "_".join(str(x) for x in item[0:-1])
                        fullpath = os.path.join(path, filename)
                        fig.savefig(f"{fullpath}.{format}")
        else:
            ## Case when which is a list of plot specified as strings
            for key in which:
                if key not in self.plots.keys():
                    print(f"Plot {key} not found in the plots dictionary")
                    print(f"All available plots are: {list(self.plots.keys())}")
                    continue
                else:
                    for item in nested_to_tuple(
                        self.plots[key]
                    ):  # Plot all epochs and splits of type key
                        fig = item[-1]  ## Figure is in last element of the tuple
                        filename = key + "_" + "_".join(str(x) for x in item[0:-1])
                        fullpath = os.path.join(path, filename)
                        fig.savefig(f"{fullpath}.{format}")



### Utilities ###

	## TODO move to BaseVisualizer
    @staticmethod
    def make_loss_format(result: Result, config: DefaultConfig) -> pd.DataFrame:
        loss_df_melt = pd.DataFrame()
        for term in result.sub_losses.keys():
            # Get the loss values and ensure it's a dictionary
            loss_values = result.sub_losses.get(key=term).get()

            # Add explicit type checking/conversion
            if not isinstance(loss_values, dict):
                # If it's not a dict, try to convert it or handle appropriately
                if hasattr(loss_values, "to_dict"):
                    loss_values = loss_values.to_dict()
                else:
                    # For non-convertible types, you might need a custom solution
                    # For numpy arrays, you could do something like:
                    if hasattr(loss_values, "shape"):
                        # For numpy arrays, create a dict with indices as keys
                        loss_values = {i: val for i, val in enumerate(loss_values)}

            # Now create the DataFrame
            loss_df = pd.DataFrame.from_dict(loss_values, orient="index")  # type: ignore

            # Rest of your code remains the same
            if term == "var_loss":
                loss_df = loss_df * config.beta
            loss_df["Epoch"] = loss_df.index + 1
            loss_df["Loss Term"] = term

            loss_df_melt = pd.concat(
                [
                    loss_df_melt,
                    loss_df.melt(
                        id_vars=["Epoch", "Loss Term"],
                        var_name="Split",
                        value_name="Loss Value",
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)

        # Similar handling for the total losses
        loss_values = result.losses.get()
        if not isinstance(loss_values, dict):
            if hasattr(loss_values, "to_dict"):
                loss_values = loss_values.to_dict()
            else:
                if hasattr(loss_values, "shape"):
                    loss_values = {i: val for i, val in enumerate(loss_values)}

        loss_df = pd.DataFrame.from_dict(loss_values, orient="index")  # type: ignore
        loss_df["Epoch"] = loss_df.index + 1
        loss_df["Loss Term"] = "total_loss"

        loss_df_melt = pd.concat(
            [
                loss_df_melt,
                loss_df.melt(
                    id_vars=["Epoch", "Loss Term"],
                    var_name="Split",
                    value_name="Loss Value",
                ),
            ],
            axis=0,
        ).reset_index(drop=True)

        loss_df_melt["Loss Value"] = loss_df_melt["Loss Value"].astype(float)
        return loss_df_melt