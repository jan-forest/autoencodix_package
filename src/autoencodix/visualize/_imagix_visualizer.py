from dataclasses import field

import matplotlib.pyplot as plt


from typing import Any, Dict, no_type_check
from autoencodix.visualize._general_visualizer import GeneralVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, get_dataset
from autoencodix.configs.default_config import DefaultConfig


class ImagixVisualizer(GeneralVisualizer):
    plots: Dict[str, Any] = field(
        default_factory=nested_dict
    )  ## Nested dictionary of plots as figure handles

    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        ## Make Model Weights plot
        ## TODO needs to be adjusted for Imagix ##
        ## Plot Model weights for each sub-VAE ##
        # self.plots["ModelWeights"] = self._plot_model_weights(model=result.model)

        ## Make long format of losses
        loss_df_melt = self._make_loss_format(result=result, config=config)

        ## Make plot loss absolute
        self.plots["loss_absolute"] = self._make_loss_plot(
            df_plot=loss_df_melt, plot_type="absolute"
        )
        ## Make plot loss relative
        self.plots["loss_relative"] = self._make_loss_plot(
            df_plot=loss_df_melt, plot_type="relative"
        )

        return result

    ## Latent space visualization via GeneralVisualizer
    # def show_latent_space(
    #     self,
    #     result: Result,
    #     plot_type: str = "2D-scatter",
    #     labels: Optional[Union[list, pd.Series, None]] = None,
    #     param: Optional[Union[list, str]] = None,
    # import pandas as pd
    #     epoch: Optional[Union[int, None]] = None,
    #     split: str = "all",
    # ) -> None:

    def show_weights(self) -> None:
        ## TODO
        raise NotImplementedError(
            "Weight visualization for X-Modalix is not implemented."
        )

    @no_type_check
    def show_image_recon_grid(self, result: Result, n_samples: int = 3) -> None:
        ## TODO add similar labels/param logic from other visualizations
        dataset = result.datasets

        ## Overwrite original datasets with new_datasets if available after predict with other data
        if dataset is None:
            dataset = DatasetContainer()

        if bool(result.new_datasets.test):
            dataset.test = result.new_datasets.test

        if dataset.test is None:
            raise ValueError("test of dataset is None")
        meta = dataset.test.metadata
        sample_ids = meta.sample(n=n_samples, random_state=42).index

        all_sample_order = dataset.test.sample_ids
        indices = [
            all_sample_order.index(sid)
            for sid in sample_ids
            if sid in all_sample_order  # ty: ignore
        ]

        fig, axes = plt.subplots(
            ncols=n_samples,  # Number of classes
            nrows=2,  # Original, Reconstructed
            figsize=(n_samples * 4, 2 * 4),
        )

        for r in range(2):
            for c in range(n_samples):
                if r == 0:
                    ## Original image
                    axes[r, c].imshow(
                        dataset.test.raw_data[indices[c]].img.squeeze()
                    )
                    axes[r, c].set_title(f"Original: {sample_ids[c]}")
                    axes[r, c].axis("off")
                if r == 1:
                    ## Reconstructed image
                    axes[r, c].imshow(
                        result.reconstructions.get(split="test", epoch=-1)[
                            indices[c]
                        ].squeeze()
                    )
                    axes[r, c].set_title(f"Reconstructed: {sample_ids[c]}")
                    axes[r, c].axis("off")

        self.plots["Image_recon_grid"] = fig
        # show_figure(fig)
        plt.show()
