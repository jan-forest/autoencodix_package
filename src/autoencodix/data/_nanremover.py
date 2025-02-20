from typing import List

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from autoencodix.data._datapackage import DataPackage


class NaNRemover:
    """Class to handle NaN removal from multi-modal data."""

    def __init__(self, relevant_cols: List[str] = None):
        """
        Initialize NaN remover.

        Parameters
        ----------
        relevant_cols : List[str], optional
            List of columns to check for NaNs in observations
        """
        self.relevant_cols = relevant_cols

    def _process_sparse_matrix(self, matrix) -> np.ndarray:
        """Convert sparse matrix to dense and ensure float type."""
        if issparse(matrix):
            matrix = matrix.toarray()
        return matrix.astype(float)

    def _remove_nan_from_matrix(
        self, matrix, name: str = ""
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove NaNs from a matrix and return valid indices."""
        matrix = self._process_sparse_matrix(matrix)
        valid_cols = ~np.isnan(matrix).any(axis=0)
        valid_rows = ~np.isnan(matrix).any(axis=1)

        if not name:
            name = "matrix"
        print(f"Found {(~valid_cols).sum()} columns with NaNs in {name}")
        print(f"Found {(~valid_rows).sum()} rows with NaNs in {name}")

        return valid_rows, valid_cols

    def _process_modality(self, adata: ad.AnnData, mod_name: str) -> ad.AnnData:
        """Process a single modality to remove NaNs."""
        print(f"\nProcessing modality: {mod_name}")
        print(f"Original shape: {adata.shape}")

        # Process X matrix
        valid_rows, valid_cols = self._remove_nan_from_matrix(adata.X, "X")
        adata = adata[valid_rows, valid_cols]

        # Process each layer
        for layer_name, layer_data in adata.layers.items():
            layer_valid_rows, layer_valid_cols = self._remove_nan_from_matrix(
                layer_data, f"layer {layer_name}"
            )
            # Update AnnData object with filtered layer
            layer_matrix = self._process_sparse_matrix(layer_data)
            adata = adata[layer_valid_rows, :]
            adata.layers[layer_name] = layer_matrix[layer_valid_rows, :][
                :, layer_valid_cols
            ]

        # Process obs annotations
        for col in self.relevant_cols:
            if col in adata.obs.columns:
                valid_obs = ~adata.obs[col].isna()
                if (~valid_obs).any():
                    print(
                        f"Removing {(~valid_obs).sum()} rows with NaNs in obs column {col}"
                    )
                    adata = adata[valid_obs, :]

        print(f"Final shape: {adata.shape}")
        return adata

    def remove_nan(self, data: DataPackage) -> DataPackage:
        """
        Remove NaNs from all components of the DataPackage.

        Parameters
        ----------
        data : DataPackage
            Data package containing multi-modal data

        Returns
        -------
        DataPackage
            Processed data package with NaNs removed
        """
        # Handle bulk data
        if data.multi_bulk:
            for key, df in data.multi_bulk.items():
                print(f"\nProcessing bulk data: {key}")
                print(f"Original shape: {df.shape}")
                data.multi_bulk[key] = df.dropna(axis=1)
                print(f"New shape: {df.shape}")

        # Handle annotation data
        if data.annotation is not None:
            print("\nProcessing annotation data")
            print(f"Original shape: {data.annotation.shape}")
            for col in self.relevant_cols:
                if col in data.annotation.columns:
                    data.annotation.dropna(subset=[col], inplace=True)
            print(f"New shape: {data.annotation.shape}")

        # Handle MuData
        if data.multi_sc is not None:
            print("\nProcessing MuData object")
            # Process each modality
            for mod_name, mod_data in data.multi_sc.mod.items():
                processed_mod = self._process_modality(mod_data, mod_name)
                data.multi_sc.mod[mod_name] = processed_mod

            # Ensure cell alignment across modalities
            common_cells = list(
                set.intersection(
                    *(set(mod.obs_names) for mod in data.multi_sc.mod.values())
                )
            )
            print(f"\nFound {len(common_cells)} common cells across all modalities")
            data.multi_sc = data.multi_sc[common_cells]

        return data
