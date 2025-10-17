from typing import Any, Dict, Optional, Tuple, Union

import mudata as md
import pandas as pd
import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._image_dataset import ImageDataset
from autoencodix.data._imgdataclass import ImgData
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.datapackage import DataPackage
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._multimodal_dataset import MultiModalDataset


class XModalPreprocessor(GeneralPreprocessor):
    """Preprocessor for cross-modal data, handling multiple data types and their transformations.


    Attributes:
        data_config: Configuration specific to data handling.
        dataset_dicts: Dictionary holding datasets for different splits (train, test, valid).
    """

    def __init__(
        self, config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None
    ):
        """Initializes the XModalPreprocessor
        Args:
            config: Configuration object for the preprocessor.
            ontologies: Optional ontologies for data processing.
        """
        super().__init__(config=config, ontologies=ontologies)
        self.data_config = config.data_config

    def preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> DatasetContainer:
        """Preprocess the data according to the configuration.
        Args:
            raw_user_data: Optional raw data provided by the user.
            predict_new_data: Flag indicating if new data is being predicted.
        """
        self.dataset_dicts = self._general_preprocess(
            raw_user_data=raw_user_data, predict_new_data=predict_new_data
        )
        datasets = {}
        for split in ["train", "test", "valid"]:
            cur_split = self.dataset_dicts.get(split)
            if cur_split is None:
                print(f"split is None: {split}")
                continue
            cur_data = cur_split.get("data")
            if not isinstance(cur_data, DataPackage):
                raise TypeError(
                    f"expected type of cur_data to be DataPackage, got {type(cur_data)}"
                )
            cur_indices = cur_split.get("indices")
            datasets[split] = MultiModalDataset(
                datasets=self._process_dp(dp=cur_data, indices=cur_indices),
                config=self.config,
            )

        for k, v in self.dataset_dicts.items():
            print(f"key: {k}, type: {type(v)}")

        return DatasetContainer(
            train=datasets["train"], test=datasets["test"], valid=datasets["valid"]
        )

    def format_reconstruction(self, reconstruction, result=None):
        pass

    def _process_dp(self, dp: DataPackage, indices: Dict[str, Any]):
        """Processes a DataPackage into a dictionary of BaseDataset objects.

        Args:
            dp: The DataPackage to process.
            indices: The indices for splitting the data.
        Returns:
            A dictionary mapping modality names to BaseDataset objects.
        """

        dataset_dict: Dict[str, BaseDataset] = {}
        for k, v in dp:
            dp_key, sub_key = k.split(".")
            data = v
            metadata = None
            if dp.annotation is not None:  # prevents error in SingleCell case
                metadata = dp.annotation.get(sub_key)
                if metadata is None:
                    metadata = dp.annotation.get("paired")
                # case where we have the unpaired case, but we have one metadata that included all samples across all numeric data
                if metadata is None:
                    if not len(dp.annotation.keys()) == 1:
                        raise ValueError(
                            f"annotation key needs to be either 'paired' match a key of the numeric data or only one key exists that holds all unpaired data, please adjust config, got: {dp.annotation.keys()}"
                        )
                    metadata_key = next(iter(dp.annotation.keys()))
                    metadata = dp.annotation.get(metadata_key)

            if dp_key == "multi_bulk":
                if not isinstance(data, pd.DataFrame):
                    raise ValueError(
                        f"Expected data for multi_bulk: {k}, {v} to be pd.DataFrame, got {type(data)}"
                    )
                if metadata is None:
                    raise ValueError("metadata cannot be None")
                metadata_num = metadata.loc[
                    data.index
                ]  # needed when we have only one annotation df containing metadata for all modalities
                dataset_dict[k] = NumericDataset(
                    data=data.values,
                    config=self.config,
                    sample_ids=data.index,
                    feature_ids=data.columns,
                    split_indices=indices,
                    metadata=metadata_num,
                )
            elif dp_key == "img":
                if not isinstance(data, list):
                    raise ValueError()
                if not isinstance(data[0], ImgData):
                    raise ValueError()
                dataset_dict[k] = ImageDataset(
                    data=data,
                    config=self.config,
                    split_indices=indices,
                    metadata=metadata,
                )
            elif dp_key == "multi_sc":
                if not isinstance(data, md.MuData):
                    raise ValueError()
                for mod_key, mod_data in data.mod.items():
                    selected_layers = self.config.data_config.data_info[
                        mod_key
                    ].selected_layers
                    if not selected_layers[0] == "X" and len(selected_layers) != 1:
                        raise NotImplementedError(
                            "Xmodalix works only with X layer of single cell data as of now"
                        )
                    dataset_dict[k] = NumericDataset(
                        data=mod_data.X,
                        config=self.config,
                        sample_ids=mod_data.obs_names,
                        feature_ids=mod_data.var_names,
                        split_indices=indices,
                        metadata=mod_data.obs,
                    )

            elif dp_key == "annotation":
                pass

            else:
                raise NotImplementedError(
                    f"Got datapackage attribute: {k}, probably you have added an attribute to the Datapackage class without adjusting this method. Only supports: ['multi_bulk', 'multi_sc', 'img' and 'annotation']"
                )
        return dataset_dict
