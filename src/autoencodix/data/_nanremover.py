from typing import List, Union

import anndata as ad  # type: ignore
import pandas as pd
import mudata as md  # type: ignore
import numpy as np
from scipy.sparse import issparse  # type: ignore

from autoencodix.data.datapackage import DataPackage


class NaNRemover:
    """Class to handle NaN removal from multi-modal data."""

    def __init__(
        self,
        relevant_cols: Union[
            List[str], None
        ] = [],  # Changed from None to empty list as default
    ) -> None:
        """
        Initialize NaNRemover with data and optional relevant columns.

        Parameters
        ----------
        data : Union[pd.DataFrame, AnnData, md.MuData]
            Data to process
        relevant_cols : List[str], optional
            Columns to check for NaN values, by default empty list
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

        return valid_rows, valid_cols

    def _process_modality(self, adata: ad.AnnData, mod_name: str) -> ad.AnnData:
        """Process a single modality to remove NaNs."""

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
        if self.relevant_cols is None:
            return adata
        for col in self.relevant_cols:
            if col in adata.obs.columns:
                valid_obs = ~adata.obs[col].isna()
                if (~valid_obs).any():
                    adata = adata[valid_obs, :]

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
                data.multi_bulk[key] = df.dropna(axis=1)

        # Handle annotation data
        if data.annotation is not None:
            non_na = {}
            for k, v in data.annotation.items():
                if v is None:
                    continue
                if self.relevant_cols is not None:
                    for col in self.relevant_cols:
                        if col in v.columns:
                            v.dropna(subset=[col], inplace=True)
                non_na[k] = v
            data.annotation = non_na  # type: ignore

        # Handle MuData in multi_sc
        if data.multi_sc is not None:
            mudata = data.multi_sc["multi_sc"]
            # Process each modality
            for mod_name, mod_data in mudata.mod.items():
                processed_mod = self._process_modality(mod_data, mod_name)
                data.multi_sc["multi_sc"].mod[mod_name] = processed_mod

            # Ensure cell alignment across modalities
            common_cells = list(
                set.intersection(*(set(mod.obs_names) for mod in mudata.mod.values()))
            )
            data.multi_sc = data.multi_sc["multi_sc"][common_cells]

        # Handle from_modality and to_modality (for translation cases)
        for direction in ["from_modality", "to_modality"]:
            modality_dict = getattr(data, direction)
            if not modality_dict:
                continue

            for mod_key, mod_value in modality_dict.items():
                # Handle MuData objects - use the proper import
                if isinstance(mod_value, md.MuData):
                    # Process each modality in the MuData
                    for inner_mod_name, inner_mod_data in mod_value.mod.items():
                        processed_mod = self._process_modality(
                            inner_mod_data, f"{mod_key}.{inner_mod_name}"
                        )
                        mod_value.mod[inner_mod_name] = processed_mod

                    # Ensure cell alignment if there are multiple modalities
                    if len(mod_value.mod) > 1:
                        common_cells = list(
                            set.intersection(
                                *(set(mod.obs_names) for mod in mod_value.mod.values())
                            )
                        )
                        mod_value = mod_value[common_cells]

                    modality_dict[mod_key] = mod_value

                # Handle AnnData objects directly
                elif isinstance(mod_value, ad.AnnData):
                    processed_mod = self._process_modality(mod_value, mod_key)
                    modality_dict[mod_key] = processed_mod

                # Handle other types of data (e.g., dictionaries of AnnData objects)
                elif isinstance(mod_value, dict):
                    for sub_key, sub_value in mod_value.items():
                        if isinstance(sub_value, ad.AnnData):
                            processed_mod = self._process_modality(
                                sub_value, f"{mod_key}.{sub_key}"
                            )
                            mod_value[sub_key] = processed_mod

                elif isinstance(mod_value, pd.DataFrame):
                    mod_value.dropna(axis=1, inplace=True)
                    modality_dict[mod_key] = mod_value

                else:
                    print(
                        f"Skipping unknown type in {direction}.{mod_key}: {type(mod_value)}"
                    )

        return data
