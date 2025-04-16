from typing import List, Dict, Union, Optional, Tuple, Any, TYPE_CHECKING

import mudata as md  # type: ignore
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore

from autoencodix.data._filter import DataFilter
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils.default_config import DataInfo

if TYPE_CHECKING:
    import mudata as md  # type: ignore

    MuData = md.MuData.MuData
else:
    MuData = Any


class SingleCellFilter:
    """Filter and scale single-cell data, returning a MuData object with synchronized metadata."""

    def __init__(
        self, data_info: Union[Dict[str, DataInfo], DataInfo], config: DefaultConfig
    ):
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
        self.data_info = data_info
        self.total_features = config.k_filter
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

    def presplit_processing(self, mudata: MuData) -> MuData:
        """
        Preprocess the data using modality-specific configurations.
        Returns
        -------
        MuData
            Preprocessed data
        """

        for mod_key, mod_data in mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is not None:
                sc.pp.filter_cells(mod_data, min_genes=data_info.min_genes)
                layers_to_process = self._get_layers_for_modality(mod_key, mod_data)

                for layer in layers_to_process:
                    if layer == "X":
                        if data_info.log_transform:
                            sc.pp.log1p(mod_data)
                    else:
                        temp_view = mod_data.copy()
                        temp_view.X = mod_data.layers[layer].copy()
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

    def sc_postsplit_processing(
        self, mudata: MuData, gene_map: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[MuData, Dict[str, List[str]]]:
        """
        Preprocess the data using modality-specific configurations.

        Parameters
        ----------
        gene_filter_dict : Optional[dict[str, list[str]]]
            Dictionary mapping modality keys to the list of genes to keep.
            If a modality key is not provided or its value is None,
            genes are filtered using `min_cells`.

        Returns
        -------
        Tuple[MuData, Dict[str, List[str]]]
            Preprocessed MuData and a dictionary mapping modality keys to the names of kept genes.
        """
        kept_genes: Dict[str, List] = {}

        for mod_key, mod_data in mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is None:
                raise ValueError(f"No data info found for modality {mod_key}")
            gene_list = gene_map.get(mod_key) if gene_map else None
            if gene_list is not None:
                gene_mask = mod_data.var_names.isin(gene_list)
                mod_data._inplace_subset_var(gene_mask)
            else:
                sc.pp.filter_genes(mod_data, min_cells=data_info.min_cells)

            kept_genes[mod_key] = mod_data.var_names.tolist()

            # Process layers with a unified loop and simplified conditions
            layers_to_process = self._get_layers_for_modality(mod_key, mod_data)
            for layer in layers_to_process:
                if layer == "X":
                    if data_info.normalize_counts:
                        sc.pp.normalize_total(mod_data)
                else:
                    # Create a temporary view to process a specific layer
                    temp_view = mod_data.copy()
                    temp_view.X = mod_data.layers[layer].copy()
                    if data_info.normalize_counts:
                        sc.pp.normalize_total(temp_view)
                    mod_data.layers[layer] = temp_view.X.copy()

            mudata.mod[mod_key] = mod_data
        mudata.update_var()

        return mudata, kept_genes

    def _apply_general_filtering(
        self, df: pd.DataFrame, data_info: DataInfo, gene_list: Optional[List]
    ) -> Tuple[pd.DataFrame, List]:
        data_processor = DataFilter(data_info=data_info)
        return data_processor.filter(df=df, genes_to_keep=gene_list)

    def _apply_scaling(
        self, df: pd.DataFrame, data_info: DataInfo, scaler: Any
    ) -> Tuple[pd.DataFrame, Any]:
        data_processor = DataFilter(data_info=data_info)
        if scaler is None:
            scaler = data_processor.fit_scaler(df=df)
        scaled_df = data_processor.scale(df=df, scaler=scaler)
        return scaled_df, scaler

    def general_postsplit_processing(
        self,
        mudata: MuData,
        gene_map: Optional[Dict[str, List]],
        scaler_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[MuData, Dict[str, List], Dict[str, Dict[str, Any]]]:
        """
        Process the single-cell data by preprocessing, filtering, and scaling.
        Returns
        -------
        mudata.MuData
            Processed MuData object with filtered and scaled values, and synchronized metadata
        """
        # At the beginning of the general_postsplit_processing method:
        feature_distribution = self.distribute_features_across_modalities(
            mudata, self.total_features
        )
        out_gene_map: Dict[str, List] = {}
        out_scaler_map: Dict[str, Dict[str, Any]] = {
            mod_key: {} for mod_key in mudata.mod.keys()
        }  # Initialize scaler map for each modality

        for mod_key, mod_data in mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            data_info.k_filter = feature_distribution[mod_key]

            gene_list = gene_map.get(mod_key) if gene_map else None

            if data_info is None:
                raise ValueError(f"No data info found for modality {mod_key}")

            layers_to_process = self._get_layers_for_modality(mod_key, mod_data)
            df = self._to_dataframe(mod_data, layer=None)  # Get DataFrame for .X
            filtered_df, gene_list = self._apply_general_filtering(
                df=df, gene_list=gene_list, data_info=data_info
            )
            out_gene_map[mod_key] = gene_list

            if scaler_map is not None:
                scaler = scaler_map[mod_key].get("X")
            else:
                scaler = None
            scaled_df, scaler = self._apply_scaling(
                df=filtered_df, data_info=data_info, scaler=scaler
            )
            out_scaler_map[mod_key]["X"] = scaler

            updated_mod_data = self._from_dataframe(
                scaled_df, mod_data, layer=None
            )  # Update .X
            mudata.mod[mod_key] = updated_mod_data
            mudata.mod[mod_key].var_names = list(
                filtered_df.columns
            )

            for layer in layers_to_process:
                if layer == "X":
                    continue  # X already processed

                if layer not in mod_data.layers:
                    raise ValueError(f"Layer '{layer}' not found in modality '{mod_key}'.")

                current_mod_data = mudata.mod[mod_key].copy()
                temp_view = current_mod_data.copy()
                temp_view.X = current_mod_data.layers[layer].copy()
                df = self._to_dataframe(mod_data=temp_view, layer=None)
                filtered_df, _ = self._apply_general_filtering(
                    df=df,
                    gene_list=gene_list,  # Use the same gene_list as for .X
                    data_info=data_info,
                )  # dont update gene_map here, because we havae the same genes as in X

                if scaler_map is not None:
                    scaler = scaler_map[mod_key].get(layer)
                else:
                    scaler = None
                scaled_df, scaler = self._apply_scaling(
                    df=filtered_df, data_info=data_info, scaler=scaler
                )
                out_scaler_map[mod_key][layer] = scaler
                scaled_view = self._from_dataframe(
                    df=scaled_df, mod_data=temp_view, layer=None
                )
                mudata.mod[mod_key].layers[layer] = scaled_view.X.copy()
                mudata.mod[mod_key].layers[layer].var_names = list(
                    filtered_df.columns
                )  # Update layer var_names
            mudata.update_var()

        return mudata, out_gene_map, out_scaler_map

    def distribute_features_across_modalities(
        self, mudata: MuData, total_features: int
    ) -> Dict[str, int]:
        """
        Distributes a total number of features across modalities evenly.

        Parameters
        ----------
        mudata : MuData
            Multi-modal data object
        total_features : int
            Total number of features to distribute across all modalities

        Returns
        -------
        Dict[str, int]
            Dictionary mapping modality keys to number of features to keep
        """
        # Count valid modalities (those that are not None)
        valid_modalities = [key for key in mudata.mod.keys()]
        n_modalities = len(valid_modalities)

        if n_modalities == 0:
            return {}

        # Calculate base features per modality
        base_features = total_features // n_modalities
        # Calculate remainder to distribute
        remainder = total_features % n_modalities

        # Distribute features
        feature_distribution = {}
        for i, mod_key in enumerate(valid_modalities):
            # Add one extra feature to early modalities if there's remainder
            extra = 1 if i < remainder else 0
            feature_distribution[mod_key] = base_features + extra

            # Set k_filter in data_info if available
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is not None:
                if not hasattr(data_info, "k_filter"):
                    setattr(data_info, "k_filter", feature_distribution[mod_key])
                else:
                    data_info.k_filter = feature_distribution[mod_key]

        return feature_distribution
