import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Literal, no_type_check

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
    
    
    def show_latent_corr(
        self,
        result: Result,
        show_table: bool = False,
        show_corrmat: bool = False
    ) -> None:
    
        rows = []
        stored_epochs = [e for e in result.latentspaces._data.keys() if e != -1]

        for epoch in stored_epochs:
            for split in ["train", "valid"]:
                latent_df = result.get_latent_df(epoch=epoch, split=split)

                if latent_df.empty:
                    continue

                latent = latent_df.select_dtypes(include=[np.number])
                latent = latent.loc[:, latent.var(axis=0) > 1e-12]

                corr = latent.corr().to_numpy()

                # upper triangle (because corr matrices are symmetric), excluding diagonal (k=1)
                tri = corr[np.triu_indices_from(corr, k=1)]

                rows.append({
                    "epoch": epoch + 1,
                    "split": split,
                    "mean_abs_pairwise_corr": np.nanmean(np.abs(tri)),
                    "max_abs_pairwise_corr": np.nanmax(np.abs(tri)),
                    "n_samples": latent.shape[0],
                    "n_latent_dims": latent.shape[1],
                })

        df_corr = pd.DataFrame(rows)

        ## Plot 1 - Mean
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_corr,
            x="epoch",
            y="mean_abs_pairwise_corr",
            hue="split",
            marker="o",
        )
        plt.title("Mean absolute pairwise correlation between latent dimensions")
        plt.xlabel("Epoch")
        plt.ylabel("Mean absolute pairwise correlation")
        plt.tight_layout()
        plt.show()

        ## Plot 2 - Max
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_corr,
            x="epoch",
            y="max_abs_pairwise_corr",
            hue="split",
            marker="o",
        )
        plt.title("Maximum absolute pairwise correlation between latent dimensions")
        plt.xlabel("Epoch")
        plt.ylabel("Max absolute pairwise correlation")
        plt.tight_layout()
        plt.show()
        
        if show_corrmat:
            # Use the final stored train/valid epoch.
            final_epoch = max(stored_epochs)
            display_epoch = final_epoch + 1

            corr_mats = {}

            for split in ["train", "valid"]:
                latent_df = result.get_latent_df(epoch=final_epoch, split=split)

                if latent_df.empty:
                    continue

                latent = latent_df.select_dtypes(include=[np.number])
                latent = latent.loc[:, latent.var(axis=0) > 1e-12]

                if latent.shape[1] < 2:
                    continue

                corr_mats[split] = latent.corr()

            if len(corr_mats) == 0:
                raise ValueError(
                    f"No valid latent correlation matrix could be computed "
                    f"for final epoch {display_epoch}."
                )

            n_plots = len(corr_mats)

            fig, axes = plt.subplots(
                nrows=1,
                ncols=n_plots,
                figsize=(7 * n_plots, 6),
                squeeze=False,
                constrained_layout=True,
            )

            for ax, (split, corr_mat) in zip(axes.ravel(), corr_mats.items()):
                sns.heatmap(
                    corr_mat,
                    ax=ax,
                    cmap="vlag",
                    vmin=-1,
                    vmax=1,
                    center=0,
                    square=True,
                    linewidths=0.0,
                    cbar=True,
                    cbar_kws={"label": "Pearson correlation"},
                    xticklabels=8,
                    yticklabels=8,
                )

                ax.set_title(
                    f"{split.capitalize()} split\n"
                    f"latent correlation matrix, epoch {display_epoch}",
                    fontsize=12,
                )
                ax.set_xlabel("Latent dimension")
                ax.set_ylabel("Latent dimension")

            fig.suptitle(
                f"Pairwise correlations between latent dimensions at epoch {display_epoch}",
                fontsize=14,
                fontweight="bold",
            )

            self.plots["LatentCorrelationMatrix"] = fig
            plt.show()
        
        # Print results table (using display for notebook, if possible)
        if show_table:
            try:
                from IPython.display import display
                display(df_corr)
            except ImportError:
                print(df_corr.to_string(index=False))
    
    
    def show_latent_activity(
        self,
        result: Result,
        final_epoch: Optional[int] = None,
        include_test: bool = True,
        show_table: bool = False,
        show_dim_table: bool = False,
        log_activity: bool = True,
    ) -> None:
        """
        Visualize latent-dimension activity statistics computed by Imagix3DEvaluator.

        This method expects that the evaluator method

        compute_latent_activity(...)

        has already been called. The evaluator stores two tables in
        result.sub_results:

        "latent_activity_summary"
        "latent_activity_by_dim"

        The visualizer then creates three diagnostic plots:

        1. Number of active latent dimensions across training epochs.
        2. Activity per latent dimension at the final selected epoch.
        3. Mean KL contribution per latent dimension at the final selected epoch,
            if KL values are available.

        Args:
            result:
                Result object containing latent-activity tables in result.sub_results.
            final_epoch:
                Stored epoch key to use for the per-dimension final-epoch plots.
                If None, the largest non-test epoch in the summary table is used.
            include_test:
                Whether to include the final test prediction rows in the
                final-epoch per-dimension plots, if they are available.
            show_table:
                If True, display the summary table.
            show_dim_table:
                If True, display the final-epoch per-dimension table.
            log_activity:
                If True, use a log scale for the activity-per-dimension plot.
                This is often helpful because latent-dimension variances can be
                highly skewed.
        """

        summary_key = "latent_activity_summary"
        dim_key = "latent_activity_by_dim"

        # if not hasattr(result, "sub_results") or result.sub_results is None:
        #     raise ValueError(
        #         "No sub_results found in result. "
        #         "Please run result = evaluator.compute_latent_activity(result) first."
        #     )

        if summary_key not in result.sub_results or dim_key not in result.sub_results:
            raise ValueError(
                "Latent activity tables were not found in result.sub_results. "
                "Please run:\n\n"
                "    result = imagix3d_loaded.evaluator.compute_latent_activity(\n"
                "        result=imagix3d_loaded.result\n"
                "    )\n\n"
                "before calling show_latent_activity()."
            )

        summary_df = result.sub_results[summary_key].copy()
        dim_df = result.sub_results[dim_key].copy()

        if summary_df.empty or dim_df.empty:
            raise ValueError(
                "Latent activity tables are empty. "
                "Please check whether compute_latent_activity() produced valid output."
            )

        # Ensure predictable order.
        summary_df = summary_df.sort_values(["epoch", "split"]).reset_index(drop=True)
        dim_df = dim_df.sort_values(["epoch", "split", "latent_dim"]).reset_index(drop=True)


        # Optional table output
        if show_table:
            try:
                from IPython.display import display
                display(summary_df)
            except ImportError:
                print(summary_df.to_string(index=False))


        # Plot 1: active units over training epochs
        train_valid_summary = summary_df.loc[~summary_df["is_test_prediction"].astype(bool)].copy()

        if not train_valid_summary.empty:
            fig, ax = plt.subplots(figsize=(8, 5))

            sns.lineplot(
                data=train_valid_summary,
                x="epoch_display",
                y="n_active_units",
                hue="split",
                marker="o",
                ax=ax,
            )

            n_latent_dims = int(train_valid_summary["n_latent_dims"].max())
            threshold = train_valid_summary["threshold"].iloc[0]

            ax.set_title("Active latent dimensions over training")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Number of active latent dimensions")
            ax.set_ylim(0, n_latent_dims + 1)

            ax.text(
                0.01,
                -0.18,
                f"Active unit criterion: Var\u2093(\u03bc\u2c7c(x)) > {threshold:g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

            fig.tight_layout()
            self.plots["LatentActivityActiveUnits"] = fig
            plt.show()
        else:
            warnings.warn(
                "No train/valid latent-activity rows found. "
                "Skipping active-units-over-training plot."
            )


        # Select final epoch for per-dimension plots
        if final_epoch is None:
            if train_valid_summary.empty:
                raise ValueError(
                    "Cannot infer final_epoch because no train/valid rows are available. "
                    "Please pass final_epoch explicitly."
                )
            final_epoch = int(train_valid_summary["epoch"].max())

        final_epoch_rows = dim_df.loc[(dim_df["epoch"] == final_epoch) & (~dim_df["is_test_prediction"].astype(bool))].copy()

        if include_test:
            test_rows = dim_df.loc[dim_df["is_test_prediction"].astype(bool)].copy()
            if not test_rows.empty:
                final_epoch_rows = pd.concat(
                    [final_epoch_rows, test_rows],
                    axis=0,
                    ignore_index=True,
                )

        if final_epoch_rows.empty:
            raise ValueError(
                f"No per-dimension latent-activity rows found for final_epoch={final_epoch}."
            )

        display_epoch = final_epoch_rows.loc[final_epoch_rows["epoch"] == final_epoch, "epoch_display"]

        if len(display_epoch) > 0:
            display_epoch = int(display_epoch.iloc[0])
        else:
            display_epoch = final_epoch + 1

        if show_dim_table:
            try:
                from IPython.display import display
                display(final_epoch_rows)
            except ImportError:
                print(final_epoch_rows.to_string(index=False))


        # Plot 2: activity per latent dimension at final epoch
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            data=final_epoch_rows,
            x="latent_dim",
            y="activity",
            hue="split",
            style="active",
            s=60,
            ax=ax,
        )

        threshold = final_epoch_rows["threshold"].iloc[0]
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            label=f"activity threshold = {threshold:g}",
        )

        ax.set_title(
            f"Latent-dimension activity at epoch {display_epoch}"
        )
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Activity: variance of posterior mean")

        if log_activity:
            positive_activity = final_epoch_rows.loc[
                final_epoch_rows["activity"] > 0, "activity"
            ]

            if not positive_activity.empty:
                ax.set_yscale("log")
            else:
                warnings.warn(
                    "All activity values are zero or non-positive. "
                    "Keeping linear y-axis for activity plot."
                )

        ax.legend(title="Split / active", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        self.plots["LatentActivityPerDimension"] = fig
        plt.show()


        # Plot 3: KL contribution per latent dimension at final epoch
        if "mean_kl" in final_epoch_rows.columns:
            kl_plot_df = final_epoch_rows.loc[np.isfinite(final_epoch_rows["mean_kl"])].copy()

            if not kl_plot_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))

                sns.lineplot(
                    data=kl_plot_df,
                    x="latent_dim",
                    y="mean_kl",
                    hue="split",
                    marker="o",
                    ax=ax,
                )

                ax.set_title(
                    f"Mean KL contribution per latent dimension at epoch {display_epoch}"
                )
                ax.set_xlabel("Latent dimension")
                ax.set_ylabel("Mean KL contribution")

                fig.tight_layout()

                self.plots["LatentActivityKLPerDimension"] = fig
                plt.show()
            else:
                warnings.warn(
                    "No finite mean_kl values found. "
                    "Skipping KL-per-dimension plot."
                )
        else:
            warnings.warn(
                "Column 'mean_kl' not found in latent-activity table. "
                "Skipping KL-per-dimension plot."
            )
            
    def show_latent_traversal(
        self,
        result: Result,
        view: Literal["axial", "coronal", "sagittal"] = "axial",
        channel: int = 0,
    ) -> None:
        """
        Visualize latent traversal volumes.

        This method expects that the evaluator method
            compute_latent_traversal(...)
        has already been called. 
        
        The evaluator stores the decoded traversal volumes and metadata in:
            result.sub_results["latent_traversal"]

        The figure layout is:
            rows    = latent dimensions
            columns = traversal values

        Args:
            result:
                Result object containing the latent traversal output in
                result.sub_results["latent_traversal"].

            view:
                Anatomical plane to visualize.
                Options:
                    "axial":    slice along the depth axis
                    "coronal":  slice along the height axis
                    "sagittal": slice along the width axis

            channel:
                Image channel to visualize. For single-channel 3D images, use 0.
        """

        traversal_key = "latent_traversal"

        if not hasattr(result, "sub_results") or result.sub_results is None:
            raise ValueError(
                "No sub_results found in result. "
                "Please run compute_latent_traversal() first."
            )

        if traversal_key not in result.sub_results:
            raise ValueError(
                "Latent traversal results were not found in result.sub_results. "
                "Please run:\n\n"
                "    result = imagix3d_loaded.evaluator.compute_latent_traversal(\n"
                "        result=imagix3d_loaded.result\n"
                "    )\n\n"
                "before calling show_latent_traversal()."
            )

        traversal = result.sub_results[traversal_key]

        if not isinstance(traversal, dict):
            raise TypeError(
                f"Expected result.sub_results[{traversal_key!r}] to be a dict, "
                f"got {type(traversal)}."
            )

        if "decoded" not in traversal:
            raise ValueError(
                f"result.sub_results[{traversal_key!r}] does not contain 'decoded'."
            )

        if "metadata" not in traversal:
            raise ValueError(
                f"result.sub_results[{traversal_key!r}] does not contain 'metadata'."
            )

        decoded = np.asarray(traversal["decoded"])
        metadata = traversal["metadata"].copy()
        settings = traversal.get("settings", {})

        if metadata.empty:
            raise ValueError("Latent traversal metadata table is empty.")

        required_columns = {
            "row_index",
            "latent_dim",
            "latent_dim_label",
            "value_index",
            "traversal_value",
            "value_mode",
        }

        missing_columns = required_columns.difference(metadata.columns)

        if missing_columns:
            raise ValueError(
                "Latent traversal metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if decoded.shape[0] != len(metadata):
            raise ValueError(
                "Number of decoded traversal volumes does not match metadata rows: "
                f"decoded.shape[0]={decoded.shape[0]}, len(metadata)={len(metadata)}."
            )

        if view not in {"axial", "coronal", "sagittal"}:
            raise ValueError(
                f"Unknown view={view!r}. "
                "Expected 'axial', 'coronal', or 'sagittal'."
            )

        # Preserve the evaluator's latent-dimension order if available.
        if isinstance(settings, dict) and "latent_dims" in settings:
            latent_dim_order = [int(dim) for dim in settings["latent_dims"]]
        else:
            latent_dim_order = (
                metadata["latent_dim"]
                .drop_duplicates()
                .astype(int)
                .tolist()
            )

        value_indices = (
            metadata["value_index"]
            .drop_duplicates()
            .astype(int)
            .sort_values()
            .tolist()
        )

        n_rows = len(latent_dim_order)
        n_cols = len(value_indices)

        if n_rows == 0 or n_cols == 0:
            raise ValueError(
                "Could not infer latent dimensions or traversal values from metadata."
            )

        fig_width = max(2.2 * n_cols, 6)
        fig_height = max(2.2 * n_rows, 4)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
            constrained_layout=True,
        )

        # Use a common intensity range for all decoded traversal volumes.
        # This makes differences across traversal values visually comparable.
        image_min = np.nanpercentile(decoded, 1)
        image_max = np.nanpercentile(decoded, 99)

        if not np.isfinite(image_min) or not np.isfinite(image_max):
            image_min = np.nanmin(decoded)
            image_max = np.nanmax(decoded)

        if image_min == image_max:
            image_min = None
            image_max = None

        for row_idx, latent_dim in enumerate(latent_dim_order):
            for col_idx, value_index in enumerate(value_indices):
                row_match = metadata.loc[
                    (metadata["latent_dim"].astype(int) == int(latent_dim))
                    & (metadata["value_index"].astype(int) == int(value_index))
                ]

                ax = axes[row_idx, col_idx]

                if row_match.empty:
                    ax.axis("off")
                    continue

                row = row_match.iloc[0]
                decoded_index = int(row["row_index"])

                volume = decoded[decoded_index]

                slice_2d = self._get_middle_slice_from_decoded_volume(
                    volume=volume,
                    view=view,
                    channel=channel,
                )

                ax.imshow(
                    slice_2d,
                    cmap="gray",
                    origin="lower",
                    vmin=image_min,
                    vmax=image_max,
                )

                ax.axis("off")

                if row_idx == 0:
                    traversal_value = float(row["traversal_value"])
                    ax.set_title(f"{traversal_value:.3g}", fontsize=10)

                if col_idx == 0:
                    mean_kl = row.get("mean_kl", np.nan)

                    if np.isfinite(mean_kl):
                        row_label = (
                            f"{row['latent_dim_label']}\n"
                            f"KL={mean_kl:.3g}"
                        )
                    else:
                        row_label = f"{row['latent_dim_label']}"

                    ax.annotate(
                        row_label,
                        xy=(-0.12, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="right",
                        fontsize=10,
                        rotation=0,
                    )

        value_mode = (
            settings.get("value_mode", metadata["value_mode"].iloc[0])
            if isinstance(settings, dict)
            else metadata["value_mode"].iloc[0]
        )

        base_sample_index = (
            settings.get("base_sample_index", None)
            if isinstance(settings, dict)
            else None
        )

        title = (
            f"Latent traversal "
            f"({view}, {value_mode} values)"
        )

        if base_sample_index is not None:
            title += f"\nBase sample index: {base_sample_index}"

        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
        )

        self.plots["LatentTraversal"] = fig

        plt.show()
    
    @staticmethod
    def _get_middle_slice_from_decoded_volume(
        volume: np.ndarray,
        view: Literal["axial", "coronal", "sagittal"],
        channel: int = 0,
    ) -> np.ndarray:
        """
        Extract the middle 2D slice from one decoded 3D volume.

        Args:
            volume:
                One decoded traversal volume.
            view:
                Anatomical plane to extract.
            channel:
                Channel index used when volume has shape (C, D, H, W).

        Returns:
            A 2D NumPy array.
        """

        volume = np.asarray(volume)

        if volume.ndim == 4:
            if channel < 0 or channel >= volume.shape[0]:
                raise ValueError(
                    f"channel={channel} is out of range for volume with "
                    f"{volume.shape[0]} channels."
                )

            volume_3d = volume[channel]

        elif volume.ndim == 3:
            volume_3d = volume

        else:
            raise ValueError(
                "Expected decoded volume to have shape (C, D, H, W) or (D, H, W), "
                f"got shape {volume.shape}."
            )

        if view == "axial":
            d_mid = volume_3d.shape[0] // 2
            return volume_3d[d_mid, :, :]

        if view == "coronal":
            h_mid = volume_3d.shape[1] // 2
            return volume_3d[:, h_mid, :]

        if view == "sagittal":
            w_mid = volume_3d.shape[2] // 2
            return volume_3d[:, :, w_mid]

        raise ValueError(
            f"Unknown view={view!r}. "
            "Expected 'axial', 'coronal', or 'sagittal'."
        )