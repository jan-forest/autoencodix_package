from typing import Any, Literal, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
import torch

from autoencodix.utils._result import Result
from autoencodix.evaluate._general_evaluator import GeneralEvaluator


class Imagix3DEvaluator(GeneralEvaluator):
    def __init__(self):
        super().__init__() # just in case GeneralEvaluator should receive new attributes, then we would want to initialize them
        

    def compute_latent_activity(
        self,
        result: Result,
        threshold: float = 0.01, # value taken from https://proceedings.neurips.cc/paper_files/paper/2021/file/6c19e0a6da12dc02239312f151072ddd-Paper.pdf
        splits: tuple[str, ...] = ("train", "valid"),
        include_test: bool = True,
    ) -> Result:
        """
        Latent-space diagnostics.

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

        result.sub_results["latent_activity_summary"] = summary_df
        result.sub_results["latent_activity_by_dim"] = dim_df

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
    
    def compute_latent_traversal(
        self,
        result: Result,
        latent_dims: Literal["all", "top_kl"] | Sequence[int] = "top_kl",
        n_latent_dims: int = 8,
        value_mode: Literal["prior", "percentile"] = "percentile",
        n_values: int = 5,
        prior_range: tuple[float, float] = (-2.0, 2.0),
        percentile_range: tuple[float, float] = (5.0, 95.0),
    ) -> Result:
        """
        Compute latent traversal.

        Latent traversal varies one latent dimension at a time while keeping all
        other latent dimensions fixed at the posterior mean of one selected sample.

        Args:
            result:
                Result object containing the trained model and stored latent means.

            latent_dims:
                Which latent dimensions to traverse.
                Options:
                    "all": traverse all latent dimensions.
                    "top_kl": traverse the n_latent_dims dimensions with highest mean KL.
                Sequence[int]:
                    explicitly provided latent dimension indices.

            n_latent_dims:
                Number of dimensions to use when latent_dims="top_kl".
                Must be between 1 and the total number of latent dimensions.

            value_mode:
                How traversal values are chosen.
                Options: "prior": use evenly spaced absolute values over prior_range.
                "percentile": use empirical percentiles of mu_j across samples.

            n_values:
                Number of traversal values per latent dimension.

            prior_range:
                Minimum and maximum absolute values for prior-based traversal.

            percentile_range:
                Lower and upper percentile for data-adaptive traversal values.

            decode_batch_size:
                Number of modified latent vectors decoded at once.

        Returns:
            The modified Result object.
    """

        if n_values < 2:
            raise ValueError("n_values must be >= 2 for a meaningful traversal.")

        mu = self._get_training_dynamic_array(
            dynamic=result.mus,
            epoch=-1,
            split="test",
            name="mu",
            allow_missing=False,
        )

        if mu is None:
            raise ValueError(
                "No posterior means found!."
        )

        n_samples, n_total_latent_dims = mu.shape

        if n_samples == 0:
            raise ValueError("No samples found for the last epoch, split='test'.")
      
        # Select a random mu vector from samples
        rng = np.random.default_rng(seed=42)
        base_sample_index = int(rng.integers(low=0, high=n_samples))
        base_z = mu[base_sample_index].astype(np.float32, copy=True)

        mean_kl = self._compute_mean_kl_for_epoch_split(
            result=result,
            mu=mu,
        )

        selected_latent_dims = self._select_latent_dims_for_traversal(
            latent_dims=latent_dims,
            n_latent_dims=n_latent_dims,
            n_total_latent_dims=n_total_latent_dims,
            mean_kl=mean_kl,
        )

        traversal_vectors: list[np.ndarray] = []
        metadata_rows: list[dict[str, Any]] = []
        values_by_dim: dict[int, list[float]] = {}

        row_index = 0

        for latent_dim in selected_latent_dims:
            traversal_values = self._make_traversal_values(
                mu=mu,
                latent_dim=latent_dim,
                value_mode=value_mode,
                n_values=n_values,
                prior_range=prior_range,
                percentile_range=percentile_range,
            )

            values_by_dim[int(latent_dim)] = [float(v) for v in traversal_values]

            for value_index, value in enumerate(traversal_values):
                z_modified = base_z.copy()
                z_modified[latent_dim] = float(value)

                traversal_vectors.append(z_modified)

                metadata_rows.append(
                    {
                        "row_index": row_index,
                        "epoch": -1,
                        "split": "test",
                        "base_sample_index": int(base_sample_index),
                        "latent_dim": int(latent_dim),
                        "latent_dim_label": f"LatDim_{latent_dim}",
                        "value_index": int(value_index),
                        "traversal_value": float(value),
                        "value_mode": value_mode,
                        "mean_kl": float(mean_kl[latent_dim])
                        if np.isfinite(mean_kl[latent_dim])
                        else np.nan,
                    }
                )

                row_index += 1

        z_traversal = np.stack(traversal_vectors, axis=0).astype(np.float32)
        decode_batch_size = int(result.model.config.batch_size)

        decoded = self._decode_latent_vectors(
            model=result.model,
            z=z_traversal,
            decode_batch_size=decode_batch_size,
        )

        metadata_df = pd.DataFrame(metadata_rows)

        if not hasattr(result, "sub_results") or result.sub_results is None:
            result.sub_results = {}

        result.sub_results["latent_traversal"] = {
            "decoded": decoded,
            "metadata": metadata_df,
            "settings": {
                "split": "test",
                "epoch": -1,
                "sample_index": int(base_sample_index),
                "latent_dims": [int(d) for d in selected_latent_dims],
                "n_total_latent_dims": int(n_total_latent_dims),
                "value_mode": value_mode,
                "n_values": int(n_values),
                "prior_range": tuple(float(v) for v in prior_range),
                "percentile_range": tuple(float(v) for v in percentile_range),
                "decode_batch_size": int(decode_batch_size),
            },
            "base_z": base_z,
            "mean_kl": mean_kl,
            "values_by_dim": values_by_dim,
        }

        return result
    
    def _compute_mean_kl_for_epoch_split(
        self,
        result: Result,
        mu: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mean KL contribution per latent dimension for one the last epoch and test split.

        Uses result.sigmas as latent log-variance, consistent with the existing
        compute_latent_activity() implementation.
        """

        n_latent_dims = mu.shape[1]

        if not hasattr(result, "sigmas") or result.sigmas is None:
            warnings.warn(
                "No result.sigmas found. Cannot compute KL-based latent ranking. "
                "Returning NaN mean KL values."
            )
            return np.full(n_latent_dims, np.nan, dtype=np.float64)

        logvar = self._get_training_dynamic_array(
            dynamic=result.sigmas,
            epoch=-1,
            split="test",
            name="logvar",
            allow_missing=True,
        )

        if logvar is None:
            warnings.warn(
                "No logvar found for last epoch, split='test'."
                "Cannot compute KL-based latent ranking."
                "Returning NaN mean KL values."
            )
            return np.full(n_latent_dims, np.nan, dtype=np.float64)

        if logvar.shape != mu.shape:
            raise ValueError(
                "mu and logvar shape mismatch for the last epoch, split='test': "
                f"mu.shape={mu.shape}, logvar.shape={logvar.shape}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            kl_per_sample_dim = -0.5 * (1.0 + logvar - mu**2 - np.exp(logvar))

        kl_per_sample_dim = np.where(
            np.isfinite(kl_per_sample_dim),
            kl_per_sample_dim,
            np.nan,
        )

        mean_kl = np.nanmean(kl_per_sample_dim, axis=0)

        return mean_kl.astype(np.float64, copy=False)
    
    @staticmethod
    def _select_latent_dims_for_traversal(
        latent_dims: Literal["all", "top_kl"] | Sequence[int],
        n_latent_dims: int,
        n_total_latent_dims: int,
        mean_kl: np.ndarray,
    ) -> list[int]:
        """
        Select latent dimensions for traversal.
        """

        if isinstance(latent_dims, str):
            if latent_dims not in {"all", "top_kl"}:
                raise ValueError(
                    f"Unknown latent_dims={latent_dims!r}. "
                    "Expected 'all', 'top_kl', or a sequence of integers."
                )

            if latent_dims == "all":
                return list(range(n_total_latent_dims))

            if n_latent_dims < 1 or n_latent_dims > n_total_latent_dims:
                raise ValueError(
                    f"n_latent_dims must be between 1 and {n_total_latent_dims}, "
                    f"got {n_latent_dims}."
                )

            if mean_kl.shape[0] != n_total_latent_dims:
                raise ValueError(
                    f"mean_kl has length {mean_kl.shape[0]}, but expected "
                    f"{n_total_latent_dims}."
                )

            if np.all(~np.isfinite(mean_kl)):
                raise ValueError(
                    "Cannot select latent_dims='top_kl' because all mean_kl values "
                    "are NaN. Run prediction with stored logvars or pass explicit "
                    "latent dimension indices."
                )

            # NaNs are treated as very small so they are sorted to the end.
            ranking_values = np.where(np.isfinite(mean_kl), mean_kl, -np.inf)

            selected = np.argsort(ranking_values)[::-1][:n_latent_dims]

            return [int(dim) for dim in selected]

        selected = [int(dim) for dim in latent_dims]

        if len(selected) == 0:
            raise ValueError("Explicit latent_dims sequence must not be empty.")

        invalid = [
            dim for dim in selected
            if dim < 0 or dim >= n_total_latent_dims
        ]

        if invalid:
            raise ValueError(
                f"Invalid latent dimension indices {invalid}. "
                f"Valid range is 0 to {n_total_latent_dims - 1}."
            )

        return selected
    
    @staticmethod
    def _make_traversal_values(
        mu: np.ndarray,
        latent_dim: int,
        value_mode: Literal["prior", "percentile"],
        n_values: int,
        prior_range: tuple[float, float],
        percentile_range: tuple[float, float],
    ) -> np.ndarray:
        """
        Create absolute replacement values for one latent dimension.
        """

        if value_mode == "prior": 
            low, high = prior_range

            if low >= high:
                raise ValueError(
                    f"prior_range must be increasing, got {prior_range}."
                )

            return np.linspace(low, high, n_values, dtype=np.float64)

        if value_mode == "percentile":
            low_pct, high_pct = percentile_range

            if low_pct < 0 or high_pct > 100 or low_pct >= high_pct:
                raise ValueError(
                    "percentile_range must satisfy "
                    f"0 <= low < high <= 100, got {percentile_range}."
                )

            percentiles = np.linspace(low_pct, high_pct, n_values)

            return np.percentile(
                mu[:, latent_dim],
                q=percentiles,
            ).astype(np.float64, copy=False)

        raise ValueError(
            f"Unknown value_mode={value_mode!r}. "
            "Expected 'prior' or 'percentile'."
        )
        
    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Unwrap common model wrappers if needed.
        """

        if hasattr(model, "_forward_module"):
            return model._forward_module

        if hasattr(model, "module"):
            return model.module

        return model
    
    def _decode_latent_vectors(
        self,
        model: torch.nn.Module,
        z: np.ndarray,
        decode_batch_size: int,
    ) -> np.ndarray:
        """
        Decode a matrix of latent vectors into reconstructed 3D volumes.

        Args:
            model:
                Trained VAE model. Must expose a decode() method.

            z:
                NumPy array of shape (n_traversal_vectors, latent_dim).

            decode_batch_size:
                Number of latent vectors decoded at once.

        Returns:
            NumPy array of decoded volumes.
            Expected shape is usually:
                (n_traversal_vectors, C, D, H, W)
        """

        if model is None:
            raise ValueError(
                "result.model is None. Latent traversal requires a trained model."
            )

        model = self._unwrap_model(model)

        if not hasattr(model, "decode"):
            raise ValueError(
                "The stored result.model does not expose a decode() method. "
                "Latent traversal requires a VAE-style model with decode()."
            )

        model.eval()

        try:
            device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError(
                "Could not determine model device because the model has no parameters."
            ) from exc

        decoded_batches: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, z.shape[0], decode_batch_size):
                stop = min(start + decode_batch_size, z.shape[0])

                z_batch = torch.as_tensor(
                    z[start:stop],
                    dtype=torch.float32,
                    device=device,
                )

                decoded = model.decode(x=z_batch)

                decoded_tensor = self._extract_decoded_tensor(decoded)

                decoded_batches.append(
                    decoded_tensor.detach().cpu().numpy()
                )

        decoded_array = np.concatenate(decoded_batches, axis=0)

        return decoded_array
    
    @staticmethod
    def _extract_decoded_tensor(decoded: Any) -> torch.Tensor:
        """
        Convert the output of model.decode(...) into a tensor.

        The expected case is that decode() returns a torch.Tensor. This helper is
        slightly defensive in case a future implementation returns an object with a
        reconstruction attribute.
        """

        if isinstance(decoded, torch.Tensor):
            return decoded

        if hasattr(decoded, "reconstruction"):
            reconstruction = decoded.reconstruction

            if isinstance(reconstruction, torch.Tensor):
                return reconstruction

        if isinstance(decoded, dict) and "reconstruction" in decoded:
            reconstruction = decoded["reconstruction"]

            if isinstance(reconstruction, torch.Tensor):
                return reconstruction

        raise TypeError(
            "model.decode(...) did not return a torch.Tensor or an object/dict "
            "with a tensor-valued 'reconstruction'. "
            f"Got type {type(decoded)}."
        )