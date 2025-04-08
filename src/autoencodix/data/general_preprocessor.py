from typing import Any, Dict, List, Optional
import torch
import numpy as np
import pandas as pd
from scipy.sparse import issparse  # type: ignore
import mudata as md  # type: ignore
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.utils.default_config import DefaultConfig


class GeneralPreprocessor(BasePreprocessor):
    """
    General Preprocessor class that uses the general_preprocessing steps from BasePreprocessor.
    It takes the split, cleaned, scaled, and filtered data packages and transforms them into a PyTorch
    Dataset that can be used in training. This class is primarily used for the Vanillix and Varix
    pipelines for numeric data.

    Attributes
    ----------
    _datapackage : Dict[str, Any]
        The processed data package containing split, cleaned, scaled, and filtered data.

    Methods
    -------
    preprocess()
        Executes the general preprocessing steps and returns the processed data package.
    _extract_primary_data(modality_data: Any) -> np.ndarray
        Extracts the primary data matrix (.X) from a modality and converts it to a dense array if sparse.
    _combine_modality_data(mudata: MuData) -> np.ndarray
        Combines the primary data matrices (.X) from all modalities in a MuData object.
    _create_numeric_dataset(data: np.ndarray, config: Any, split_ids: np.ndarray, metadata: Any, ids: List[str]) -> NumericDataset
        Creates a NumericDataset from the given data and metadata.
    _process_multi_bulk(data_dict: Dict[str, pd.DataFrame], config: Any, split_ids: np.ndarray, metadata: Any) -> NumericDataset
        Processes multi-bulk data by concatenating all dataframes and creating a NumericDataset.
    _process_multi_sc(mudata: MuData, config: Any, split_ids: np.ndarray, metadata: Any) -> NumericDataset
        Processes multi-single-cell data by combining modalities and creating a NumericDataset.
    _process_data_package(data_dict: Dict[str, Any]) -> BaseDataset
        Processes a data package based on its type (multi-bulk or multi-single-cell).
    """

    def __init__(self, config: DefaultConfig):
        """
        Initializes the GeneralPreprocessor with the given configuration.
        self.feature_ids_dict = feature_ids_dict or {}

        Parameters:
            config (DefaultConfig): Configuration for the preprocessor.
        """
        super().__init__(config=config)
        self._datapackage: Optional[Dict[str, Any]] = None

    def _extract_primary_data(self, modality_data: md.MuData) -> np.ndarray:
        """
        Extracts the primary data matrix (.X) from a modality and converts it to a dense array if sparse.

        Parameters:
            modality_data (Any): The modality data (e.g., AnnData object).

        Returns:
            np.ndarray: The primary data matrix as a dense array.
        """
        primary_data = modality_data.X
        if issparse(primary_data):
            primary_data = primary_data.toarray()
        return primary_data

    def _combine_layers(
        self, modality_name: str, modality_data: Any
    ) -> List[np.ndarray]:
        layer_list: List[np.ndarray] = []
        selected_layers = self.config.data_config.data_info[
            modality_name
        ].selected_layers

        for layer_name in selected_layers:
            if layer_name == "X":
                primary_data = self._extract_primary_data(modality_data)
                layer_list.append(primary_data)
            elif layer_name in modality_data.layers:
                # Handle additional layers
                layer_data = modality_data.layers[layer_name]
                # Convert sparse matrix to dense if necessary
                if issparse(layer_data):
                    layer_data = layer_data.toarray()
                layer_list.append(layer_data)
            else:
                print(
                    f"Layer '{layer_name}' not found in modality '{modality_name}'. Skipping."
                )
        return layer_list

    def _combine_modality_data(self, mudata: md.MuData) -> np.ndarray:
        """
        Combines the primary data matrices (.X) and specified layers from all modalities in a MuData object.

        Parameters:
            mudata (MuData): The MuData object containing multiple modalities.

        Returns:
            np.ndarray: The combined data matrix.
        """
        modality_data_list = []

        for modality_name, modality_data in mudata.mod.items():
            combined_layers = self._combine_layers(
                modality_name=modality_name, modality_data=modality_data
            )
            modality_data_list.extend(combined_layers)

        return np.concatenate(modality_data_list, axis=1)

    def _create_numeric_dataset(
        self,
        data: np.ndarray,
        config: DefaultConfig,
        split_ids: np.ndarray,
        metadata: pd.DataFrame,
        ids: List[str],
        feature_ids: List[str],
    ) -> NumericDataset:
        """
        Creates a NumericDataset from the given data and metadata.

        Parameters:
            data (np.ndarray): The data matrix.
            config (Any): Configuration for the dataset.
            split_ids (np.ndarray): Indices for splitting the data.
            metadata (Any): Metadata associated with the data.
            ids (List[str]): Identifiers for the observations.

        Returns:
            NumericDataset: The created NumericDataset.
        """
        tensor_data = torch.from_numpy(data)
        return NumericDataset(
            data=tensor_data,
            config=config,
            split_ids=split_ids,
            metadata=metadata,
            ids=ids,
            feature_ids=feature_ids,
        )

    def _process_data_package(self, data_dict: Dict[str, Any]) -> BaseDataset:
        """
        Processes a data package based on its type (multi-bulk or multi-single-cell).

        Parameters:
            data_dict (Dict[str, Any]): The data package containing data and metadata.

        Returns:
            BaseDataset: The created NumericDataset.

        Raises:
            ValueError: If no data is found in the split.
        """
        data, split_ids = data_dict["data"], data_dict["indices"]

        for key in data.__annotations__.keys():
            attr_val = getattr(data, key)
            if key == "multi_bulk" and attr_val is not None:
                metadata = data.annotation
                dfs_to_concat = list(attr_val.values())

                combined_cols = []
                for df in dfs_to_concat:
                    if isinstance(df, pd.DataFrame):
                        combined_cols.extend(df.columns)
                    else:
                        raise ValueError(
                            "Expected a DataFrame, but got something else."
                        )

                combined_df = pd.concat(dfs_to_concat, axis=1)

                return self._create_numeric_dataset(
                    data=combined_df.values,
                    config=self.config,
                    split_ids=split_ids,
                    metadata=metadata,
                    ids=combined_df.index.tolist(),
                    feature_ids=combined_cols,
                )

            elif key == "multi_sc" and attr_val is not None:
                combined_data = self._combine_modality_data(attr_val)
                combined_obs = pd.concat(
                    [modality_data.obs for modality_data in attr_val.mod.values()],
                    axis=1,
                )
                return self._create_numeric_dataset(
                    data=combined_data,
                    config=self.config,
                    split_ids=split_ids,
                    metadata=combined_obs,
                    ids=attr_val.obs_names.tolist(),
                    feature_ids=attr_val.var_names.tolist(),
                )

        raise NotImplementedError(f"General Preprocessor is not implemented for {key}")

    def preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> DatasetContainer:
        """
        Executes the general preprocessing steps and returns the processed data package.

        Returns:
            Dict[str, Any]: The processed data package.
        """

        self.predict_new_data = predict_new_data
        self._datapackage = self._general_preprocess(
            raw_user_data=raw_user_data, predict_new_data=predict_new_data
        )
        print(self._datapackage)
        if self._datapackage is None:
            raise TypeError("Datapackage cannot be None")
        self._dataset_container = DatasetContainer()
        for split in ["train", "test", "valid"]:
            if self._datapackage[split]["data"] is None:
                self._dataset_container[split] = None
                continue
            dataset = self._process_data_package(data_dict=self._datapackage[split])
            self._dataset_container[split] = dataset

        return self._dataset_container
