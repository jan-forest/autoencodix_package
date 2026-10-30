from dataclasses import dataclass
from typing import Dict, Literal, TypeAlias
import numpy as np

ScalingMethod: TypeAlias = Literal["STANDARD", "MINMAX", "ROBUST", "NONE"]

@dataclass
class GlobalVolumeNormalizer:
    """
    Normalizes (scales) 3D images using stats fitted on the training set.

    This class implements train-global normalization for volumetric image data.
    Stats are computed once across all training images and then reused for transforming
    training, validation, and test data.

    Supported normalization methods are:
        - STANDARD: subtract channel-wise mean and divide by channel-wise standard deviation
        - MINMAX: subtract channel-wise minimum and divide by channel-wise range
        - ROBUST: subtract channel-wise median and divide by channel-wise interquartile range
        - NONE: no normalization

    Attributes:
        method: The normalization method to apply.
        normalize_nonzero_only: If True, compute statistics only on nonzero voxels
        stats: Dictionary containing fitted channel-wise normalization statistics.
        The exact entries depend on the selected method:
            - MINMAX: "min", "max"
            - STANDARD: "mean", "std"
            - ROBUST: "median", "iqr"
    """
    method: ScalingMethod
    normalize_nonzero_only: bool
    stats: Dict[str, np.ndarray]
    
    @classmethod
    def fit(
        cls,
        images: list,
        method: ScalingMethod,
        normalize_nonzero_only: bool,
        ) -> "GlobalVolumeNormalizer":
        """
        Fits a GlobalVolumeNormalizer on a list of training images.

        This class method computes normalization statistics across all training images 
        and returns a fitted normalizer object.

        Args:
            images: List of image data objects.
            method: The normalization method to fit. 
            normalize_nonzero_only: If True, fit statistics only on nonzero voxels.

        Returns:
            A fitted GlobalVolumeNormalizer instance containing the selected method
            and the corresponding channel-wise statistics.

        Raises:
            ValueError: If an unsupported normalization method is provided.
        """
        SUPPORTED_METHODS = {"STANDARD", "MINMAX", "ROBUST", "NONE"}
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        if method == "NONE":
            return cls(
                method="NONE",
                normalize_nonzero_only=normalize_nonzero_only,
                stats={}
            )

        eps = np.finfo(np.float32).eps

        # collect all train voxels
        arrays = [img.img.astype(np.float32, copy=False) for img in images]
        stacked = np.stack(arrays, axis=0)   # (N, C, D, H, W)

        if normalize_nonzero_only:
            mask = stacked != 0
        else:
            mask = np.ones_like(stacked, dtype=bool)

        stats = {}
        c = stacked.shape[1]

        if method == "MINMAX":
            mins, maxs = [], []
            for ch in range(c):
                vals = stacked[:, ch][mask[:, ch]]
                mins.append(vals.min() if vals.size else 0.0)
                maxs.append(vals.max() if vals.size else 1.0)
            stats["min"] = np.asarray(mins, dtype=np.float32)
            stats["max"] = np.asarray(maxs, dtype=np.float32)

        elif method == "STANDARD":
            means, stds = [], []
            for ch in range(c):
                vals = stacked[:, ch][mask[:, ch]]
                means.append(vals.mean() if vals.size else 0.0)
                stds.append(vals.std() if vals.size else 1.0)
            stats["mean"] = np.asarray(means, dtype=np.float32)
            stats["std"] = np.asarray(stds, dtype=np.float32) + eps

        elif method == "ROBUST":
            medians, iqrs = [], []
            for ch in range(c):
                vals = stacked[:, ch][mask[:, ch]]
                if vals.size:
                    q75, q25 = np.percentile(vals, [75, 25])
                    medians.append(np.median(vals))
                    iqrs.append((q75 - q25) + eps)
                else:
                    medians.append(0.0)
                    iqrs.append(1.0)
            stats["median"] = np.asarray(medians, dtype=np.float32)
            stats["iqr"] = np.asarray(iqrs, dtype=np.float32)


        return cls(
            method=method,
            normalize_nonzero_only=normalize_nonzero_only,
            stats=stats,
        )
    
    def transform_volume(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the fitted normalization transform to a single 3D image volume.
        The transformation uses the statistics previously computed by `fit()`.
        
        Args:
            image: Input image volume as a NumPy array.

        Returns:
            The normalized image volume as a NumPy array.

        Raises:
            ValueError: If the stored normalization method is unsupported.
        """
        if self.method == "NONE":
            return image

        image = image.astype(np.float32, copy=True)
        eps = np.finfo(np.float32).eps

        if self.normalize_nonzero_only:
            for c in range(image.shape[0]):
                channel = image[c]
                mask = channel != 0
                if not np.any(mask):
                    continue

                vals = channel[mask]
                if self.method == "MINMAX":
                    vmin = self.stats["min"][c]
                    vmax = self.stats["max"][c]
                    channel[mask] = (vals - vmin) / ((vmax - vmin) + eps)
                elif self.method == "STANDARD":
                    mean = self.stats["mean"][c]
                    std = self.stats["std"][c]
                    channel[mask] = (vals - mean) / (std + eps)
                elif self.method == "ROBUST":
                    median = self.stats["median"][c]
                    iqr = self.stats["iqr"][c]
                    channel[mask] = (vals - median) / (iqr + eps)

                image[c] = channel
            return image

        # normalize all voxels
        if self.method == "MINMAX":
            return (image - self.stats["min"][:, None, None, None]) / (
                (self.stats["max"] - self.stats["min"])[:, None, None, None] + eps
            )
        elif self.method == "STANDARD":
            return (image - self.stats["mean"][:, None, None, None]) / (
                self.stats["std"][:, None, None, None] + eps
            )
        elif self.method == "ROBUST":
            return (image - self.stats["median"][:, None, None, None]) / (
                self.stats["iqr"][:, None, None, None] + eps
            )

        raise ValueError(f"Unsupported method: {self.method}")