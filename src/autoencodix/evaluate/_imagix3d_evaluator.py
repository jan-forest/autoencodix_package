from typing import Optional
import warnings

import numpy as np
import pandas as pd

from autoencodix.utils._result import Result
from autoencodix.evaluate._general_evaluator import GeneralEvaluator


class Imagix3DEvaluator(GeneralEvaluator):
    def __init__(self):
        super().__init__() # just in case GeneralEvaluator should receive new attributes, then we would want to initialize them
        pass

    def compute_latent_activity(
        self,
        result: Result,
        threshold: float = 0.01, # value taken from https://proceedings.neurips.cc/paper_files/paper/2021/file/6c19e0a6da12dc02239312f151072ddd-Paper.pdf
        splits: tuple[str, ...] = ("train", "valid"),
        include_test: bool = True,
    ) -> Result:
        """
        Evaluator extension for Imagix3D-specific latent-space diagnostics.

        The latent activity diagnostic follows the common "active units" idea:
        a latent dimension is considered active if the variance of its posterior
        mean across samples exceeds a threshold.

        Conceptually:
            activity_j = Var_x(mu_j(x))

        where mu_j(x) is the encoder's posterior mean for latent dimension j.

        The active-units diagnostic is used as a proxy for latent-space utilization 
        and posterior collapse.
        """
        
        
        if not hasattr(result, "mus") or result.mus is None:
            raise ValueError(
                "No posterior means found in result.mus. "
                "Latent activity requires stored posterior means."
            )

        if not hasattr(result.mus, "_data"):
            raise ValueError(
                "result.mus does not expose a _data attribute. "
                "Cannot infer stored epochs for latent activity."
            )
        
        stored_epochs = sorted(e for e in result.mus._data.keys() if e != -1)
        summary_rows = []
        dim_rows = []
        
        if len(stored_epochs) == 0 and not include_test:
            raise ValueError(
                "No stored train/valid posterior means found in result.mus."
            )

        # Compute train/valid activity over stored training epochs.
        for epoch in stored_epochs:
            for split in splits:
                if split == "test":
                    # Test is the final prediction split, it is handled down below via epoch = -1
                    continue
                
                self._append_latent_activity_for_epoch_split(
                    result=result,
                    epoch=epoch,
                    split=split,
                    threshold=threshold,
                    summary_rows=summary_rows,
                    dim_rows=dim_rows,
                    is_test_prediction=False,
                )
        
        # Compute final test-set activity once, if requested
        if include_test:
            self._append_latent_activity_for_epoch_split(
                result=result,
                epoch=-1,
                split="test",
                threshold=threshold,
                summary_rows=summary_rows,
                dim_rows=dim_rows,
                is_test_prediction=True,
            )
        
        summary_df = pd.DataFrame(summary_rows)
        dim_df = pd.DataFrame(dim_rows)
        
        # Store results without touching the existing embedding_evaluation table.
        if not hasattr(result, "sub_results") or result.sub_results is None:
            result.sub_results = {}

        result.sub_results["imagix3d_latent_activity_summary"] = summary_df
        result.sub_results["imagix3d_latent_activity_by_dim"] = dim_df

        return result
    
    def _append_latent_activity_for_epoch_split(
        self,
        result: Result,
        epoch: int,
        split: str,
        threshold: float,
        summary_rows: list[dict],
        dim_rows: list[dict],
        is_test_prediction: bool = False,
    ) -> None:
        """
        Compute latent activity for one epoch/split and append rows.
        Mutates summary_rows and dim_rows in place.
        """
        mu = self._get_training_dynamic_array(
            dynamic=result.mus,
            epoch=epoch,
            split=split,
            name="mu",
            allow_missing=True,
        )

        logvar = None
        if hasattr(result, "sigmas") and result.sigmas is not None:
            logvar = self._get_training_dynamic_array(
                dynamic=result.sigmas,
                epoch=epoch,
                split=split,
                name="logvar",
                allow_missing=True,
            )
        n_samples, n_latent_dims = mu.shape

        if n_samples < 2:
            warnings.warn(
                f"Skipping latent activity for epoch={epoch}, split={split}: "
                f"need at least 2 samples, got {n_samples}."
            )
            return

        # Activity definition: variance of posterior mean across samples.
        mu_var = np.var(mu, axis=0)
        active = mu_var > threshold

        mean_mu = np.mean(mu, axis=0)
        std_mu = np.std(mu, axis=0)

        if logvar is not None:
            mean_logvar = np.mean(logvar, axis=0)

            with np.errstate(over="ignore", invalid="ignore"):
                kl_per_sample_dim = -0.5 * (1 + logvar - mu**2 - np.exp(logvar))

            # replaces invalid KL values with NaN
            kl_per_sample_dim = np.where(
                np.isfinite(kl_per_sample_dim),
                kl_per_sample_dim,
                np.nan,
            )

            mean_kl = np.nanmean(kl_per_sample_dim, axis=0)
        else:
            mean_logvar = np.full(n_latent_dims, np.nan)
            mean_kl = np.full(n_latent_dims, np.nan)

        n_active_units = int(np.sum(active))

        if epoch >= 0:
            epoch_display = epoch + 1
        else:
            # Use model.config.epochs as the human-readable final epoch if possible.
            epoch_display = getattr(result.model.config, "epochs", np.nan)

        summary_rows.append(
            {
                "epoch": epoch,
                "epoch_display": epoch_display,
                "split": split,
                "is_test_prediction": is_test_prediction,
                "threshold": threshold,
                "n_samples": n_samples,
                "n_latent_dims": n_latent_dims,
                "n_active_units": n_active_units,
                "fraction_active": n_active_units / n_latent_dims,
                "mean_activity": float(np.mean(mu_var)),
                "median_activity": float(np.median(mu_var)),
                "max_activity": float(np.max(mu_var)),
                "sum_kl": float(np.nansum(mean_kl)),
                "mean_kl": float(np.nanmean(mean_kl)),
            }
        )

        for dim_idx in range(n_latent_dims):
            dim_rows.append(
                {
                    "epoch": epoch,
                    "epoch_display": epoch_display,
                    "split": split,
                    "is_test_prediction": is_test_prediction,
                    "threshold": threshold,
                    "latent_dim": dim_idx,
                    "latent_dim_label": f"LatDim_{dim_idx}",
                    "activity": float(mu_var[dim_idx]),
                    "active": bool(active[dim_idx]),
                    "mean_mu": float(mean_mu[dim_idx]),
                    "std_mu": float(std_mu[dim_idx]),
                    "mean_logvar": float(mean_logvar[dim_idx]),
                    "mean_kl": float(mean_kl[dim_idx]),
                }
            )
            
            
    @staticmethod
    def _get_training_dynamic_array(
        dynamic,
        epoch: int,
        split: str,
        name: str,
        allow_missing: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Retrieve one stored TrainingDynamics array as a 2D NumPy array.

        Args:
            dynamic:
                A TrainingDynamics-like object with a .get(epoch=..., split=...)
                method.
            epoch:
                Stored epoch key.
            split:
                Dataset split name, e.g. "train", "valid", or "test".
            name:
                Human-readable name used in warnings/errors.
            allow_missing:
                If True, return None when the requested data are missing.
                If False, raise ValueError.

        Returns:
            A NumPy array with shape (n_samples, n_latent_dims), or None.
        """

        try:
            arr = dynamic.get(epoch=epoch, split=split)
        except Exception as exc:
            if allow_missing:
                warnings.warn(
                    f"Could not retrieve {name} for epoch={epoch}, split={split}. "
                    f"Skipping this entry. Original error: {exc}"
                )
                return None
            raise

        if arr is None:
            if allow_missing:
                return None
            raise ValueError(
                f"No {name} found for epoch={epoch}, split={split}."
            )

        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()

        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.size == 0:
            if allow_missing:
                return None
            raise ValueError(
                f"Empty {name} array for epoch={epoch}, split={split}."
            )

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected {name} for epoch={epoch}, split={split} to be 2D "
                f"(n_samples, n_latent_dims), got shape {arr.shape}."
            )

        return arr.astype(np.float64, copy=False)