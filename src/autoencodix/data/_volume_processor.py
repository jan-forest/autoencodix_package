from typing import List, Dict, Optional, Tuple, Union
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage
from autoencodix.data import ImgData, ImagePreprocessor
from autoencodix.utils._volreader import VolumeDataReader, VolumeNormalizer
from autoencodix.configs import DataCase, Imagix3DConfig


class VolumePreprocessor(ImagePreprocessor):
    """
    Preprocessor for cross-modal data, handling multiple data types and their transformations.


    Attributes:
        data_config: Configuration specific to data handling and preprocessing.
        config: Complete 3D image-specific configuration 
        dataset_dicts: Dictionary holding datasets for different splits (train/test/valid).
        data_readers: A dictionary mapping DataCase enum values to data reader instances for 
            different modalities. Overwritten here for 3D images
    """

    def __init__(
        self, config: Imagix3DConfig, ontologies: Optional[Union[Tuple, Dict]] = None
    ):
        super().__init__(config=config, ontologies=ontologies)
        self.data_config = config.data_config
        self.config: Imagix3DConfig = config 
        
        self.data_readers[DataCase.IMG_TO_BULK]["img"] = VolumeDataReader(config=self.config)
        self.data_readers[DataCase.SINGLE_CELL_TO_IMG]["img"] = VolumeDataReader(config=self.config)
        self.data_readers[DataCase.IMG_TO_IMG] = VolumeDataReader(config=self.config)
    
    def preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> DatasetContainer:
        """
        Preprocess the data according to the configuration.

        Args:
            raw_user_data: The raw data package provided by the user.
            predict_new_data: Flag indicating if new data is being predicted.
        Returns:
            A DatasetContainer with processed training, validation, and test datasets.
        """
        self.dataset_dicts = self._general_preprocess(
            raw_user_data=raw_user_data, predict_new_data=predict_new_data
        )
        datasets = {}
        for split in ["train", "test", "valid"]:
            cur_split = self.dataset_dicts.get(split)
            if cur_split is None:
                print(f"split is None: {split}")
                datasets[split] = None # NEWLY ADDED: fix ImagePreprocessor bug for splits later!
                continue
            
            cur_data = cur_split.get("data")
            if cur_data is None: # NEWLY ADDED: fix ImagePreprocessor bug for splits later!
                datasets[split] = None # NEWLY ADDED: fix ImagePreprocessor bug for splits later!
                continue # NEWLY ADDED: fix ImagePreprocessor bug for splits later!
            
            if not isinstance(cur_data, DataPackage):
                raise TypeError(
                    f"expected type of cur_data to be DataPackage, got {type(cur_data)}"
                )
            cur_indices = cur_split.get("indices")
            datasets[split] = self._process_dp(dp=cur_data, indices=cur_indices) #type: ignore

        return DatasetContainer(
            train=datasets["train"], test=datasets["test"], valid=datasets["valid"]
        )

    def _normalize_image_data(self, images: List[ImgData], info_key: str) -> List[ImgData]:
        """
        Process 3D images with normalization.

        Normalizes a list of 3D image data objects using VolumeNormalizer based on
        the scaling method specified in the configuration for the given info_key.

        Args:
            images: A list of image data objects (each having an 'img' attribute).
            info_key: The key referencing data information in the configuration to get the scaling method.

        Returns:
            A list of processed image data objects with normalized image data.
        """

        scaling_method = self.config.data_config.data_info[info_key].scaling
        if scaling_method == "NOTSET":
            scaling_method = self.config.scaling
        
        SUPPORTED_METHODS = {"STANDARD", "MINMAX", "ROBUST", "NONE"}
        if scaling_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Scaling method '{scaling_method}' is not supported for 2D/3D image data. "
                f"Supported methods are: {sorted(SUPPORTED_METHODS)}."
            )
                        
        processed_images = []
        for img in images:
            img.img = VolumeNormalizer.normalize_volume(
                image=img.img, 
                method=scaling_method, #type: ignore
                normalize_nonzero_only=self.config.normalize_nonzero_only
            )
            processed_images.append(img)

        return processed_images