import matplotlib.pyplot as plt
import numpy as np
from typing import no_type_check

from autoencodix.visualize._imagix_visualizer import ImagixVisualizer
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.utils._result import Result


class Imagix3DVisualizer(ImagixVisualizer):

    @no_type_check
    def show_image_recon_grid(
        self, 
        result: Result, 
        n_samples: int = 3, 
        selection: str = "random", 
        metric: str = "mse"
    ) -> None: 
        """
        Plot original, reconstructed, and absolute-error slices for test samples.

        Args:
            result: Result object.
            n_samples: Number of random samples if selection="random".
                       Ignored if selection="bmw".
            selection: "random" or "bmw".
                       "bmw" selects best, median, and worst reconstruction.
            metric: Metric used for bmw selection. Options: "mse" or "bce".
        """
        
        ## TODO add similar labels/param logic from other visualizations
        
        if selection not in ["random", "bmw"]:
            raise ValueError("selection must be either 'random' or 'bmw'.")
        
        if metric not in ["mse", "bce"]:
            raise ValueError("metric must be either 'mse' or 'bce'.")
        
        def reconstruction_score(orig, recon, metric):
            """Compute reconstruction score for sample selection"""
            if metric == "mse":
                return float(np.mean((orig - recon) ** 2))
            
            if metric == "bce":
                # bce assumes target values in [0, 1]
                # If recon is outside [0, 1] treat it as logits and apply sigmoid.
                y = np.clip(orig, 0.0, 1.0)
                
                if np.nanmin(recon) < 0.0 or np.nanmax(recon) > 1.0:
                    p= 1.0 / (1.0 + np.exp(-recon))
                else:
                    p = recon
                
                eps = np.finfo(float).eps
                p = np.clip(p, eps, 1.0 - eps)
                
                bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
                return float(np.mean(bce))
                              
                
        dataset = result.datasets

        ## Overwrite original datasets with new_datasets if available after predict with other data
        if dataset is None:
            dataset = DatasetContainer()

        if result.new_datasets.test:
            dataset.test = result.new_datasets.test

        if dataset.test is None:
            raise ValueError("test of dataset is None")
        
        meta = dataset.test.metadata
        #n_samples = min(n_samples, len(meta))
        #sample_ids = meta.sample(n=n_samples, random_state=42).index

        all_sample_order = dataset.test.sample_ids
        
        # get reconstructed images
        recons = result.reconstructions.get(split="test", epoch=-1)
        if recons is None: 
                raise ValueError("No reconstructions found in result")
        
        indices = [
            idx for idx, sid in enumerate(all_sample_order)
            if sid in meta.index and idx < len(recons)
        ]
        
        if selection == "random":
            n_samples = min(n_samples, len(meta))
            sample_ids = meta.sample(n=n_samples, random_state=42).index
            
            selected = []
            for sid in sample_ids:
                if sid in all_sample_order:
                    idx = all_sample_order.index(sid)
                    selected.append(
                        {
                            "idx": idx,
                            "sample_id": sid,
                            "label": "random",
                            "score": None
                        }
                    )
        
        else:
            # bmw: best, median and worst by reconstruction score
            scored_samples = []
            
            for idx in indices:
                sid = all_sample_order[idx]
                
                # original volume
                orig = dataset.test.raw_data[idx].img.squeeze()
                
                # reconstructed volume
                recon = recons[idx].squeeze()
            
                score = reconstruction_score(orig, recon, metric)
            
                scored_samples.append(
                    {
                        "idx": idx,
                        "sample_id": sid,
                        "score": score
                    }
                )
        
            scored_samples = sorted(scored_samples, key=lambda x: x["score"])
            
            best = scored_samples[0]    # first entry in ordered samples
            median = scored_samples[len(scored_samples) // 2]
            worst = scored_samples[-1]  # last entry in ordered samples
        
            selected =   [
                {**worst, "label": "worst"},
                {**median, "label": "median"},
                {**best, "label": "best"}
            ]
        
            n_samples = len(selected) # i.e. 3
            
        
        fig, axes = plt.subplots(
            ncols=n_samples,
            nrows=9,  # Original [ax., cor., sag.], Reconstructed [ax., cor., sag.]
            figsize=(n_samples * 4, 9 * 4),
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
            "Error\nAxial",
            "Error\nCoronal",
            "Error\nSagittal",
        ]

        for c, item in enumerate(selected):
            
            idx = item["idx"]
            sample_id = item["sample_id"]
            
            # original volume
            orig = dataset.test.raw_data[idx].img.squeeze()

            # reconstructed volume
            recon = recons[idx].squeeze()
            
            error = np.abs(orig - recon)
            
            # index for the middle slice of every dimension
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
            
            error_slices = [
                error[d_mid, :, :],  # axial
                error[:, h_mid, :],  # coronal
                error[:, :, w_mid],  # sagittal
            ]
                    
            all_slices = orig_slices + recon_slices + error_slices
            
            # Use same intensity range for orig and recon of this sample
            image_min = min(np.nanmin(orig), np.nanmin(recon))
            image_max = max(np.nanmax(orig), np.nanmax(recon))
            
            # Error map range
            error_min = 0.0
            error_max = np.nanpercentile(error, 99)
            if error_max < 0:
                error_max = np.nanmax(error)

            for r in range(9):
                
                if r < 6:
                    axes[r, c].imshow(all_slices[r], cmap="gray", origin="lower", vmin=image_min, vmax=image_max)
                else:
                    axes[r, c].imshow(all_slices[r], cmap="magma", origin="lower", vmin=error_min, vmax=error_max)
                    
                axes[r, c].axis("off")

                # title only on top row
                if r == 0:
                    if item["score"] is None:
                        axes[r, c].set_title(f"{sample_id}")
                    else:
                        axes[r, c].set_title(
                            f"{item['label']}: {sample_id}\n"
                            f"{metric.upper()} = {item['score']: .4g}"
                        )

                # row label only on first column
                if c == 0:
                    axes[r, c].annotate(
                        row_labels[r],
                        xy=(-0.15, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="right",
                        fontsize=11,
                        rotation=0,
                    )

        self.plots["Image_recon_grid"] = fig
        
        # show_figure(fig)
        plt.show()