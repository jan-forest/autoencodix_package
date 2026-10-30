from typing import List, Dict, Optional, Tuple, Union, Any, cast
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage
from autoencodix.data import ImgData, ImagePreprocessor, GlobalVolumeNormalizer
from autoencodix.utils._volreader import VolumeDataReader, VolumeNormalizer
from autoencodix.configs import DataCase, Imagix3DConfig


class VolumePreprocessor(ImagePreprocessor):
    """
    Preprocessor for 3D image data

    Attributes:
        data_config: Configuration specific to data handling and preprocessing.
        config: Complete 3D image-specific configuration 
        dataset_dicts: Dictionary holding datasets for different splits (train/test/valid).
        data_readers: A dictionary mapping DataCase enum values to data reader instances for 
            different modalities. Overwritten here for 3D images
    """

    def __init__(
        self, 
        config: Imagix3DConfig, 
        ontologies: Optional[Union[Tuple, Dict]] = None
    ):
        super().__init__(config=config, ontologies=ontologies)
        self.data_config = config.data_config
        self.config: Imagix3DConfig = config 
        self.volume_scalers: Optional[Dict[str, GlobalVolumeNormalizer]] = None
        
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
    
    def _postsplit_volume_data(
        self,
        split_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        
        """
        Fit a volume scaler on the train split only and apply it to all splits.
        This is the 3D-image analogue of _postsplit_multi_bulk().
        """
        processed_splits: Dict[str, Dict[str, Any]] = {}

        train_split = split_data.get("train")
        if train_split is None:
            raise ValueError("Train split data is None.")

        train_data = train_split.get("data")
        if train_data is None:
            raise ValueError("Train split data is None.")

        if train_data.img is None:
            raise ValueError("No img attribute found in train DataPackage.")

        if self.volume_scalers is None:
            self.volume_scalers = {}

            for modality_key, img_list in train_data.img.items():
                if img_list is None:
                    continue

                scaling_method = self.config.data_config.data_info[modality_key].scaling
                if scaling_method == "NOTSET":
                    scaling_method = self.config.scaling

                scaler = GlobalVolumeNormalizer.fit(
                    images=img_list,
                    method=scaling_method, # type: ignore
                    normalize_nonzero_only=self.config.normalize_nonzero_only,
                )
                self.volume_scalers[modality_key] = scaler

                for img in img_list:
                    img.img = scaler.transform_volume(img.img)

        else:
            for modality_key, img_list in train_data.img.items():
                if img_list is None:
                    continue
                scaler = self.volume_scalers[modality_key]
                for img in img_list:
                    img.img = scaler.transform_volume(img.img)

        processed_splits["train"] = {
            "data": train_data,
            "indices": split_data["train"]["indices"],
        }

        for split_name, split_package in split_data.items():
            if split_name == "train":
                continue

            if split_package["data"] is None:
                processed_splits[split_name] = split_package
                continue

            split_dp = split_package["data"]
            if split_dp.img is None:
                processed_splits[split_name] = split_package
                continue

            for modality_key, img_list in split_dp.img.items():
                if img_list is None:
                    continue
                scaler = self.volume_scalers[modality_key]
                for img in img_list:
                    img.img = scaler.transform_volume(img.img)

            processed_splits[split_name] = {
                "data": split_dp,
                "indices": split_package["indices"],
            }

        return processed_splits
    
    def _process_img_to_img_case(
        self,
        raw_user_data: Optional[DataPackage] = None,
    ) -> Dict[str, DataPackage]:
        """
        Signature kept identical to BasePreprocessor to avoid override complaints.
        Real implementation lives in _process_img_to_img_case_volume().
        """
        out = self._process_img_to_img_case_volume(raw_user_data=raw_user_data)
        return cast(Dict[str, DataPackage], out)

    def _process_img_to_img_case_volume(
        self,
        raw_user_data: Optional[DataPackage] = None,
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        if raw_user_data is None:
            imgreader = self.data_readers[DataCase.IMG_TO_IMG]
            images, annotation = imgreader.read_data(config=self.config)
            data_package = DataPackage(img=images, annotation=annotation)
        else:
            data_package = raw_user_data

        if self.config.requires_paired:
            common_ids = data_package.get_common_ids()
            images = data_package.img
            if images is None:
                raise ValueError("Images cannot be None")
            data_package.img = {
                k: self.filter_imgdata_list(img_list=v, ids=common_ids)
                for k, v in images.items()
            }

        if self.config.volume_scaling_strategy == "per_volume":
            def presplit_processor(modality_data: Dict[str, List[ImgData]]) -> Dict[str, List[ImgData]]:
                return {k: self._normalize_image_data(v, k) for k, v in modality_data.items()}

            def postsplit_processor(split_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
                return split_data

        elif self.config.volume_scaling_strategy == "train_global":
            def presplit_processor(modality_data: Dict[str, List[ImgData]]) -> Dict[str, List[ImgData]]:
                return modality_data

            def postsplit_processor(split_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
                return self._postsplit_volume_data(split_data=split_data)

        else:
            raise ValueError(
                f"Unsupported volume_scaling_strategy: {self.config.volume_scaling_strategy}"
            )

        return self._process_data_case(
            data_package,
            modality_processors={
                "img": (
                    presplit_processor,
                    postsplit_processor,
                ),
            },
        )
    