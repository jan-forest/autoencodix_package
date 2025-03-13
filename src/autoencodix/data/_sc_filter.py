from typing import List, Dict

import mudata as md # type: ignore
import numpy as np
import pandas as pd
import scanpy as sc # type: ignore

from autoencodix.data._filter import DataFilter
from autoencodix.utils.default_config import DataInfo


class SingleCellFilter:
    """Filter and scale single-cell data, returning a MuData object with synchronized metadata."""

    def __init__(self, mudata: md.MuData, data_info: Dict[str,DataInfo]):
        """
        Initialize single-cell filter.
        Parameters
        ----------
        mudata : MuData
            Multi-modal data to be filtered
        data_info : Union[SCDataInfo, Dict[str, SCDataInfo]]
            Either a single data_info object for all modalities or a dictionary of data_info objects
            for each modality.
        """
        self.mudata = mudata
        self.data_info = data_info
        self._is_data_info_dict = isinstance(data_info, dict)

    def _get_data_info_for_modality(self, mod_key: str) -> DataInfo:
        """
        Get the data_info configuration for a specific modality.
        Parameters
        ----------
        mod_key : str
            The modality key (e.g., "RNA", "METH")
        Returns
        -------
        SCDataInfo
            The data_info configuration for the modality
        """
        if self._is_data_info_dict:
            info = self.data_info.get(mod_key)  # type: ignore
            if info is None:
                raise ValueError(f"No data info found for modality {mod_key}")
            return info
        return self.data_info  # type: ignore

    def _get_layers_for_modality(self, mod_key: str, mod_data) -> List[str]:
        """
        Get the layers to process for a specific modality.
        Parameters
        ----------
        mod_key : str
            The modality key (e.g., "RNA", "METH")
        mod_data : AnnData
            The AnnData object for the modality
        Returns
        -------
        List[str]
            List of layer names to process. If None or empty, returns ['X'] for default layer.
        """
        data_info = self._get_data_info_for_modality(mod_key)
        selected_layers = data_info.selected_layers

        # Validate that the specified layers exist
        available_layers = list(mod_data.layers.keys())
        valid_layers = []

        for layer in selected_layers:
            if layer == "X":
                valid_layers.append("X")
            elif layer in available_layers:
                valid_layers.append(layer)
            else:
                print(
                    f"Warning: Layer '{layer}' not found in modality '{mod_key}'. Skipping."
                )
        if not valid_layers:
            valid_layers = ["X"]

        return valid_layers

    def _preprocess(self) -> md.MuData:
        """
        Preprocess the data using modality-specific configurations.
        Returns
        -------
        MuData
            Preprocessed data
        """
        mudata = self.mudata.copy()

        for mod_key, mod_data in mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is not None:
                sc.pp.filter_cells(mod_data, min_genes=data_info.min_genes)
                sc.pp.filter_genes(mod_data, min_cells=data_info.min_cells)
                layers_to_process = self._get_layers_for_modality(mod_key, mod_data)

                for layer in layers_to_process:
                    if layer == "X":
                        if data_info.normalize_counts:
                            sc.pp.normalize_total(mod_data)
                        if data_info.log_transform:
                            sc.pp.log1p(mod_data)
                    else:
                        temp_view = mod_data.copy()
                        temp_view.X = mod_data.layers[layer].copy()

                        if data_info.normalize_counts:
                            sc.pp.normalize_total(temp_view)
                        if data_info.log_transform:
                            sc.pp.log1p(temp_view)
                        mod_data.layers[layer] = temp_view.X.copy()

                mudata.mod[mod_key] = mod_data

        return mudata

    def _to_dataframe(self, mod_data, layer=None) -> pd.DataFrame:
        """
        Transform a modality's AnnData object to a pandas DataFrame.
        Parameters
        ----------
        mod_data : AnnData
            Modality data to be transformed
        layer : str or None
            Layer to convert to DataFrame. If None, uses X.
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        if layer is None or layer == "X":
            data = mod_data.X
        else:
            data = mod_data.layers[layer]

        # Convert to dense array if sparse
        if isinstance(data, np.ndarray):
            matrix = data
        else:  # Assuming it's a sparse matrix
            matrix = data.toarray()

        return pd.DataFrame(
            matrix, columns=mod_data.var_names, index=mod_data.obs_names
        )

    def _from_dataframe(self, df: pd.DataFrame, mod_data, layer=None):
        """
        Update a modality's AnnData object with the values from a DataFrame.
        This also synchronizes the `obs` and `var` metadata to match the filtered data.
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the updated values
        mod_data : AnnData
            Modality data to be updated
        layer : str or None
            Layer to update with DataFrame values. If None, updates X.
        Returns
        -------
        AnnData
            Updated AnnData object
        """
        # Filter the AnnData object to match the rows and columns of the DataFrame
        filtered_mod_data = mod_data[df.index, df.columns].copy()

        # Update the data matrix with the filtered and scaled values
        if layer is None or layer == "X":
            filtered_mod_data.X = df.values
        else:
            if layer not in filtered_mod_data.layers:
                filtered_mod_data.layers[layer] = df.values
            else:
                filtered_mod_data.layers[layer] = df.values

        return filtered_mod_data

    def apply_general_filtering_and_scaling(self, mod_data, data_info, layer=None):
        """
        Apply general filtering and scaling to a modality's data.
        Parameters
        ----------
        mod_data : AnnData
            Modality data to be filtered and scaled
        data_info : BaseModel
            Configuration for filtering and scaling
        layer : str or None
            Layer to process. If None, processes X.
        Returns
        -------
        AnnData
            Filtered and scaled modality data with synchronized metadata
        """
        df = self._to_dataframe(mod_data, layer)
        data_filter = DataFilter(df, data_info)
        filtered_df = data_filter.filter()
        scaled_df = data_filter.scale(filtered_df)
        updated_mod_data = self._from_dataframe(scaled_df, mod_data, layer)
        return updated_mod_data

    def preprocess(self) -> md.MuData:
        """
        Process the single-cell data by preprocessing, filtering, and scaling.
        Returns
        -------
        MuData
            Processed MuData object with filtered and scaled values, and synchronized metadata
        """
        preprocessed_mudata = self._preprocess()

        for mod_key, mod_data in preprocessed_mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is not None:
                # Get the layers to process
                layers_to_process = self._get_layers_for_modality(mod_key, mod_data)

                # First process X (if in the list) to create the base filtered structure
                if "X" in layers_to_process:
                    updated_mod_data = self.apply_general_filtering_and_scaling(
                        mod_data,
                        data_info,
                        layer=None,  # None means use X
                    )
                    # Update the modality with filtered structure
                    preprocessed_mudata.mod[mod_key] = updated_mod_data
                else:
                    # If X is not in the list, still need a filtered structure
                    # Create it based on the first layer in the list
                    first_layer = layers_to_process[0]
                    temp_view = mod_data.copy()

                    # Temporarily set the first layer as X
                    temp_view.X = (
                        mod_data.layers[first_layer].copy()
                        if first_layer != "X"
                        else mod_data.X
                    )

                    updated_mod_data = self.apply_general_filtering_and_scaling(
                        temp_view, data_info, layer=None
                    )

                    # Update the modality with filtered structure
                    preprocessed_mudata.mod[mod_key] = updated_mod_data

                # Now process the remaining layers
                for layer in layers_to_process:
                    if layer == "X":  # X already processed or skipped
                        continue
                        # Make sure the layer exists in the updated structure
                    if layer in mod_data.layers:
                        current_mod_data = preprocessed_mudata.mod[mod_key].copy()

                        # Apply filtering and scaling to this layer
                        temp_view = current_mod_data.copy()
                        temp_view.X = current_mod_data.layers[layer].copy()

                        filtered_view = self.apply_general_filtering_and_scaling(
                            temp_view, data_info, layer=None
                        )

                        preprocessed_mudata.mod[mod_key].layers[layer] = (
                            filtered_view.X.copy()
                        )

        return preprocessed_mudata
