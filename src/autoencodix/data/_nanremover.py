import anndata as ad  # type: ignore
import warnings
import pandas as pd
import mudata as md  # type: ignore
import numpy as np
from scipy.sparse import issparse  # type: ignore
from scipy import sparse  # type: ignore

from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.default_config import DefaultConfig


class NaNRemover:
    """Removes NaN values from multi-modal datasets.

    This object identifies and removes NaN values from various data structures
    commonly used in single-cell and multi-modal omics, including AnnData, MuData,
    and Pandas DataFrames. It supports processing of X matrices, layers, and
    observation annotations within AnnData objects, as well as handling bulk and
    annotation data within a DataPackage.

    Attributes:
        config: Configuration object containing settings for data processing.
        relevant_cols: List of columns in metadata to check for NaNs.
    """

    def __init__(
        self,
        config: DefaultConfig,
    ):
        """Initialize the NaNRemover with configuration settings.
        Args:
            config: Configuration object containing settings for data processing.

        """
        self.config = config
        self.relevant_cols = self.config.data_config.annotation_columns

    def _process_modality(self, adata: ad.AnnData) -> ad.AnnData:
        """Converts NaN values in AnnData object to zero and metadata NaNs to 'missing'.
        Args:
            adata: The AnnData object to process.
        Returns:
            The processed AnnData object with NaN values replaced.
        """
        adata = adata.copy()

        # Handle X matrix
        if sparse.issparse(adata.X):
            if hasattr(adata.X, "data"):
                adata.X.data = np.nan_to_num(  # ty:  ignore
                    adata.X.data, nan=0.0
                )  # ty: ignore[invalid-assignment]
                adata.X.eliminate_zeros()  # ty: ignore
        else:
            adata.X = np.nan_to_num(adata.X, nan=0.0)

        # Handle all layers
        for layer_name, layer_data in adata.layers.items():
            if sparse.issparse(layer_data):
                if hasattr(layer_data, "data"):
                    layer_data.data = np.nan_to_num(layer_data.data, nan=0.0)
                    layer_data.eliminate_zeros()
            else:
                adata.layers[layer_name] = np.nan_to_num(layer_data, nan=0.0)

        # Handle obs metadata
        if self.relevant_cols is not None:
            print(adata.obs.columns)
            for col in self.relevant_cols:
                if col in adata.obs.columns:
                    # Fill NaNs with "missing" for non-numeric columns
                    if not pd.api.types.is_numeric_dtype(adata.obs[col]):
                        # Add "missing" to categories first, then fill
                        adata.obs[col] = adata.obs[col].cat.add_categories(["missing"])
                        adata.obs[col] = (
                            adata.obs[col].fillna("missing").astype("category")
                        )
        return adata

    def remove_nan(self, data: DataPackage) -> DataPackage:
        """Removes NaN values from all applicable DataPackage components.

        Iterates through the bulk data, annotation data, and multi-modal
        single-cell data (MuData and AnnData objects) within the provided
        DataPackage and removes rows/columns/entries containing NaN values.

        Args:
            data: The DataPackage object containing multi-modal data.

        Returns:
            The DataPackage object with NaN values removed from its components.
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
                        # Fill with "missing" if column is not integer or float
                        if col in v.columns and not pd.api.types.is_numeric_dtype(
                            v[col]
                        ):
                            v.fillna(value={col: "missing"}, inplace=True)

                non_na[k] = v
            data.annotation = non_na  # type: ignore

        # Handle MuData in multi_sc
        if data.multi_sc is not None and self.config.requires_paired:
            mudata = data.multi_sc["multi_sc"]
            # Process each modality
            for mod_name, mod_data in mudata.mod.items():
                processed_mod = self._process_modality(adata=mod_data)
                data.multi_sc["multi_sc"].mod[mod_name] = processed_mod

        elif data.multi_sc is not None:
            print(f"data in multi_sc: {data.multi_sc}")
            processed = {k: None for k, _ in data.multi_sc.items()}

            for k, v in data.multi_sc.items():
                # we know from screader that there is only one modality
                for modkey, adata in v.mod.items():
                    processed_mod = self._process_modality(adata=adata)
                    processed_mod = md.MuData({modkey: processed_mod})
                processed[k] = processed_mod
            data.multi_sc = processed

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
                        processed_mod = self._process_modality(inner_mod_data)
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
                    processed_mod = self._process_modality(mod_value)
                    modality_dict[mod_key] = processed_mod

                # Handle other types of data (e.g., dictionaries of AnnData objects)
                elif isinstance(mod_value, dict):
                    for sub_key, sub_value in mod_value.items():
                        if isinstance(sub_value, ad.AnnData):
                            processed_mod = self._process_modality(sub_value)
                            mod_value[sub_key] = processed_mod

                elif isinstance(mod_value, pd.DataFrame):
                    mod_value.dropna(axis=1, inplace=True)
                    modality_dict[mod_key] = mod_value

                else:
                    warnings.warn(
                        f"Skipping unknown type in {direction}.{mod_key}: {type(mod_value)}"
                    )

        return data
