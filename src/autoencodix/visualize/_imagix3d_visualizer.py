
import matplotlib.pyplot as plt

from typing import no_type_check
from autoencodix.visualize._imagix_visualizer import ImagixVisualizer
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.utils._result import Result


class Imagix3DVisualizer(ImagixVisualizer):

    @no_type_check
    def show_image_recon_grid(self, result: Result, n_samples: int = 3) -> None:
        ## TODO add similar labels/param logic from other visualizations
        dataset = result.datasets

        ## Overwrite original datasets with new_datasets if available after predict with other data
        if dataset is None:
            dataset = DatasetContainer()

        if result.new_datasets.test:
            dataset.test = result.new_datasets.test

        if dataset.test is None:
            raise ValueError("test of dataset is None")
        meta = dataset.test.metadata
        n_samples = min(n_samples, len(meta))
        sample_ids = meta.sample(n=n_samples, random_state=42).index

        all_sample_order = dataset.test.sample_ids
        indices = [
            all_sample_order.index(sid)
            for sid in sample_ids
            if sid in all_sample_order  
        ]
        
        # get reconstructed images
        recons = result.reconstructions.get(split="test", epoch=-1)
        if recons is None: 
                raise ValueError("No reconstructions found in result")
        
        fig, axes = plt.subplots(
            ncols=n_samples,
            nrows=6,  # Original [ax., cor., sag.], Reconstructed [ax., cor., sag.]
            figsize=(n_samples * 4, 6 * 4),
            squeeze=False,
            constrained_layout=True
        )
        
        row_labels = [
            "Original\nAxial",
            "Original\nCoronal",
            "Original\nSagittal",
            "Recon\nAxial",
            "Recon\nCoronal",
            "Recon\nSagittal",
        ]

        for c, idx in enumerate(indices):
            
            # original volume
            orig = dataset.test.raw_data[idx].img.squeeze()

            # reconstructed volume
            recon = recons[idx].squeeze()

            d_mid, h_mid, w_mid = [s // 2 for s in orig.shape]
                               
            # extract slices
            orig_slices = [
                orig[d_mid, :, :],   # axial
                orig[:, h_mid, :],   # coronal
                orig[:, :, w_mid],   # sagittal
            ]

            recon_slices = [
                recon[d_mid, :, :],  # axial
                recon[:, h_mid, :],  # coronal
                recon[:, :, w_mid],  # sagittal
            ]
                    
            all_slices = orig_slices + recon_slices

            for r in range(6):
                axes[r, c].imshow(all_slices[r], cmap="gray")
                axes[r, c].axis("off")

                # title only on top row
                if r == 0:
                    axes[r, c].set_title(f"{sample_ids[c]}")

                # row label only on first column
                if c == 0:
                    axes[r, c].set_ylabel(row_labels[r], rotation=0, labelpad=40, va="center")

        self.plots["Image_recon_grid"] = fig
        # show_figure(fig)
        plt.show()