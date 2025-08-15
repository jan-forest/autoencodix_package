from typing import Any, Dict, Optional, Tuple, Union


from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._image_dataset import ImageDataset
from autoencodix.data.datapackage import DataPackage
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.configs.default_config import DefaultConfig


class ImagePreprocessor(GeneralPreprocessor):
    """
    Preprocessor for cross-modal data, handling multiple data types and their transformations.
    Inherits from BasePreprocessor.
    """

    def __init__(
        self, config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None
    ):
        super().__init__(config=config, ontologies=ontologies)
        self.data_config = config.data_config

    def preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> DatasetContainer:
        """
        Preprocess the data according to the configuration.
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
            datasets[split] = self._process_dp(dp=cur_data, indices=cur_indices)

        return DatasetContainer(
            train=datasets["train"], test=datasets["test"], valid=datasets["valid"]
        )

    def format_reconstruction(self, reconstruction, result=None):
        pass

    def _process_dp(self, dp: DataPackage, indices: Dict[str, Any]) -> ImageDataset:
        first_key = next(iter(list(dp.img.keys())))
        if not isinstance(dp.img, dict):
            raise TypeError(
                f"Expected `img` attribute of DataPackage to be `dict`, got {type(dp.img)}"
            )
        if len(dp.img.keys()) > 1:
            import warnings

            warnings.warn(
                f"got multiple image datasets for Imagix: {dp.img.keys()},\
                          we only support a single image dataset in this case, using: {first_key}"
            )
        if dp.annotation is None:
            metadata = None
        else:
            metadata = dp.annotation.get(first_key)
            if metadata is None:
                metadata = dp.annotation.get("paired")
        data = dp.img[first_key]
        return ImageDataset(
            data=data,
            config=self.config,
            split_indices=indices,
            metadata=metadata,
        )
