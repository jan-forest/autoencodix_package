from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from scipy.sparse import issparse  # type: ignore
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_preprocessor import BasePreprocessor
import anndata as ad
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.utils.default_config import DefaultConfig, DataCase
from autoencodix.utils._result import Result

class StackixPreprocessor(BasePreprocessor):
    """
    Preprocessor for Stackix architecture, which handles multiple modalities separately.

    Unlike GeneralPreprocessor which combines all modalities, StackixPreprocessor
    keeps modalities separate for individual VAE training in the Stackix architecture.

    Attributes
    ----------
    config : DefaultConfig
        Configuration parameters for preprocessing and model architecture
    _datapackage : Optional[Dict[str, Any]]
        Dictionary storing processed data splits
    _dataset_container : DatasetContainer
        Container for processed datasets by split
    """

    def __init__(self, config: DefaultConfig):
        """
        Initialize the StackixPreprocessor with the given configuration.

        Parameters
        ----------
        config : DefaultConfig
            Configuration parameters for preprocessing
        """
        super().__init__(config=config)
        self._datapackage: Optional[Dict[str, Any]] = None
        self._dataset_container: Optional[DatasetContainer] = None

    def preprocess(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> DatasetContainer:
        """
        Execute preprocessing steps for Stackix architecture.

        Unlike GeneralPreprocessor, this keeps modalities separate for individual VAE training.

        Parameters
        ----------
        raw_user_data : Optional[DataPackage]
            Raw user data to preprocess, or None to use self._datapackage

        Returns
        -------
        DatasetContainer
            Container with StackixDataset for each split

        Raises
        ------
        TypeError
            If datapackage is None after preprocessing
        """
        self._datapackage = self._general_preprocess(raw_user_data)
        self._dataset_container = DatasetContainer()

        for split in ["train", "valid", "test"]:
            if (
                split not in self._datapackage
                or self._datapackage[split].get("data") is None
            ):
                self._dataset_container[split] = None
                continue
            dataset_dict = self._build_dataset_dict(
                datapackage=self._datapackage[split]["data"],
                split_ids=self._datapackage[split]["indices"],
            )
            stackix_ds = StackixDataset(
                dataset_dict=dataset_dict,
                config=self.config,
                split=split,
            )
            self._dataset_container[split] = stackix_ds
        return self._dataset_container

    def _extract_primary_data(self, modality_data: Any) -> np.ndarray:
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
                data = self._extract_primary_data(modality_data)
                layer_list.append(data)
            elif layer_name in modality_data.layers:
                layer_data = modality_data.layers[layer_name]
                if issparse(layer_data):
                    layer_data = layer_data.toarray()
                layer_list.append(layer_data)
            else:
                continue
        return layer_list

    def _build_datset_dict(
        self, datapackage: DataPackage, split_indices: List[Any]
    ) -> Dict[str, NumericDataset]:
        """
        For each seperate entry in our datapackge we build a NumericDataset
        and store it in a dictionary with the modality as key.

        Parameters
        ----------
        datapackage : DataPackage
            DataPackage containing the data to be processed
        Returns
        -------
        Dict[str, NumericDataset]
            Dictionary mapping modality names to NumericDataset objects

        """
        dataset_dict: Dict[str, NumericDataset] = {}
        for k, v in datapackage:
            attr_name, dict_key = k.split(".")
            if attr_name == "multi_bulk":
                df = datapackage[attr_name][dict_key]
                tensor = torch.from_numpy(df.values)

                ds = NumericDataset(
                    data=tensor,
                    config=self.config,
                    sample_ids=df.index,
                    feature_ids=df.columns,
                    metadata=datapackage["annotation"][dict_key],
                    split_indices=split_indices,
                )
                dataset_dict[dict_key] = ds
            elif attr_name == "multi_sc":
                mudata = datapackage["multi_sc"]["multi_sc"]
                if isinstance(mudata, ad.AnnData):
                    raise TypeError(
                        "Expected a MuData object, but got an AnnData object."
                    )

                layer_list: List[Any] = []
                for mod_name, mod_data in mudata.mod.items():
                    layer_list.extend(
                        self._combine_layers(
                            modality_name=mod_name, modality_data=mod_data
                        )
                    )
                    mod_concat = np.concatenate(layer_list, axis=1)
                    ds = NumericDataset(
                        data=torch.from_numpy(mod_concat),
                        config=self.config,
                        sample_ids=mudata.obs_names,
                        feature_ids=mod_data.var_names * len(layer_list),
                        metadata=mod_data.obs,
                        split_indices=split_indices,
                    )
                    dataset_dict[mod_name] = ds
            else:
                continue
        return dataset_dict



    def format_reconstruction(self, result: Result) -> DataPackage:
        """
        Takes the reconsturcted tensor and from which modality it comes and uses the dataset_dict
        to obtain the format of the original datapackge, but instead of the .data attribute
        we populate this attribute with the reconstructed tensor (as pd.DataFrame or MuData object)
        
        Parameters
        ----------
        reconstruction : torch.Tensor
            The reconstructed tensor
        modalit_key : str
            The key of the modality in the dataset_dict
        Returns
        -------
        DataPackage
            The reconstructed data package with the reconstructed tensor
        
        """
        stackix_ds = self._dataset_container[split]
        if stackix_ds is None:
            raise ValueError(f"No dataset found for split: {split}")
        dataset_dict = stackix_ds.dataset_dict
        data_package = DataPackage()
        if self.config.data_case == DataCase.MULTI_BULK:
            for k,v in 