from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
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

    def __init__(
        self, config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None
    ) -> None:
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
                split_indices=self._datapackage[split]["indices"],
            )
            stackix_ds = StackixDataset(
                dataset_dict=dataset_dict,
                config=self.config,
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
    ) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
        """
        Combine layers from a modality and return the combined data and indices.

        Parameters
        ----------
        modality_name : str
            Name of the modality
        modality_data : Any
            Data for the modality

        Returns
        -------
        Tuple[np.ndarray, List[Tuple[str, int, int]]]
            Combined data and list of (layer_name, start_idx, end_idx) tuples
        """
        layer_list: List[np.ndarray] = []
        layer_indices: Dict[str, Tuple[int]] = {}

        selected_layers = self.config.data_config.data_info[
            modality_name
        ].selected_layers

        start_idx = 0
        print("combine layers")
        for layer_name in selected_layers:
            print(f"layer: {layer_name}")
            if layer_name == "X":
                data = self._extract_primary_data(modality_data)
                layer_list.append(data)
                end_idx = start_idx + data.shape[1]
                layer_indices[layer_name] = [start_idx, end_idx]
                start_idx += data.shape[1]
                continue
            elif layer_name in modality_data.layers:
                layer_data = modality_data.layers[layer_name]
                if issparse(layer_data):
                    layer_data = layer_data.toarray()
                layer_list.append(layer_data)
                end_idx = start_idx + layer_data.shape[1]
                layer_indices[layer_name] = [start_idx, end_idx]
                start_idx += layer_data.shape[1]

        combined_data = (
            np.concatenate(layer_list, axis=1) if layer_list else np.array([])
        )
        return combined_data, layer_indices

    def _build_dataset_dict(
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
        layer_id_dict: Dict[str : Dict[str, List]] = {}
        for k, _ in datapackage:
            attr_name, dict_key = k.split(
                "."
            )  # see DataPackage __iter__ method for why this makes sense

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
                print("building dataset_dict")
                for mod_name, mod_data in mudata.mod.items():
                    layers, indices = self._combine_layers(
                        modality_name=mod_name, modality_data=mod_data
                    )
                    layer_id_dict[mod_name] = indices
                    layer_list.append(layers)
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
        self._layer_indices = layer_id_dict
        return dataset_dict

    def format_reconstruction(
        self, reconstruction: Any, result: Optional[Result] = None
    ) -> DataPackage:
        """
        Takes the reconstructed tensor and from which modality it comes and uses the dataset_dict
        to obtain the format of the original datapackage, but instead of the .data attribute
        we populate this attribute with the reconstructed tensor (as pd.DataFrame or MuData object)

        Parameters
        ----------
        reconstruction : torch.Tensor
            The reconstructed tensor
        result : Optional[Result]
            The result object containing the reconstructed tensor and other information
        Returns
        -------
        DataPackage
            The reconstructed data package with the reconstructed tensor

        """
        if result is None:
            raise ValueError(
                "Result object is not provided. This is needed for the StackixPreprocessor."
            )
        reconstruction = result.sub_reconstructions
        if not isinstance(reconstruction, dict):
            raise TypeError(
                f"Expected value to be of type dict for Stackix, got {type(reconstruction)}."
            )

        if self.config.data_case == DataCase.MULTI_BULK:
            return self._format_multi_bulk(reconstruction=reconstruction)

        elif self.config.data_case == DataCase.MULTI_SINGLE_CELL:
            return self._format_multi_sc(reconstruction=reconstruction)

    def _format_multi_bulk(
        self, reconstruction: Dict[str, torch.Tensor]
    ) -> DataPackage:
        multi_bulk_dict = {}
        annotation_dict = {}
        dp = DataPackage()
        for name, reconstruction in reconstruction.items():
            if not isinstance(reconstruction, torch.Tensor):
                raise TypeError(
                    f"Expected value to be of type torch.Tensor, got {type(reconstruction)}."
                )
            if self._dataset_container is None:
                raise ValueError("Dataset container is not initialized.")
            stackix_ds = self._dataset_container["test"]
            if stackix_ds is None:
                raise ValueError("No dataset found for split: test")
            dataset_dict = stackix_ds.dataset_dict
            df = pd.DataFrame(
                reconstruction.numpy(),
                index=dataset_dict[name].sample_ids,
                columns=dataset_dict[name].feature_ids,
            )
            multi_bulk_dict[name] = df
            annotation_dict[name] = dataset_dict[name].metadata

        dp["multi_bulk"] = multi_bulk_dict
        dp["annotation"] = annotation_dict
        return dp

    def _format_multi_sc(self, reconstruction: Dict[str, torch.Tensor]) -> DataPackage:
        """
        Formats reconstructed tensors back into a MuData object for single-cell data.

        This uses the stored layer indices to accurately split the reconstructed tensor
        back into the original layers.

        Parameters
        ----------
        reconstruction : Dict[str, torch.Tensor]
            Dictionary of reconstructed tensors for each modality

        Returns
        -------
        DataPackage
            DataPackage containing the reconstructed MuData object
        """
        import mudata as md

        dp = DataPackage()
        modalities = {}

        if self._dataset_container is None:
            raise ValueError("Dataset container is not initialized.")
        if not hasattr(self, "_layer_indices"):
            raise ValueError(
                "Layer indices not found. Make sure _build_dataset_dict was called."
            )

        stackix_ds = self._dataset_container["test"]
        if stackix_ds is None:
            raise ValueError("No dataset found for split: test")

        dataset_dict = stackix_ds.dataset_dict

        # Process each modality in the reconstruction
        for mod_name, recon_tensor in reconstruction.items():
            if not isinstance(recon_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected value to be of type torch.Tensor, got {type(recon_tensor)}."
                )
            if mod_name not in dataset_dict:
                raise ValueError(f"Modality {mod_name} not found in dataset dictionary")
            original_dataset = dataset_dict[mod_name]

            layer_indices = self._layer_indices[mod_name]

            start_idx, end_idx = layer_indices["X"]
            x_data = recon_tensor.numpy()[:, start_idx:end_idx]

            var_names = original_dataset.feature_ids

            mod_data = ad.AnnData(
                X=x_data,
                obs=original_dataset.metadata,
                var=pd.DataFrame(index=var_names[0 : x_data.shape[1]]),
            )

            # Add additional layers based on stored indices
            for layer_name, ids in layer_indices.items():
                if layer_name == "X":
                    continue  # X is already set

                layer_data = recon_tensor.numpy()[:, ids[0] : ids[1]]
                mod_data.layers[layer_name] = layer_data

            modalities[mod_name] = mod_data

        # Create MuData object from all modalities
        mdata = md.MuData(modalities)

        # Create and return DataPackage
        dp["multi_sc"] = {"multi_sc": mdata}
        return dp
