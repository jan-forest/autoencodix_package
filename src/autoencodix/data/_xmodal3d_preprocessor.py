from typing import Any, Dict, List, Optional, Tuple, Union, cast

from autoencodix.configs.default_config import DataCase
from autoencodix.configs.xmodalix3d_config import XModalix3DConfig
from autoencodix.configs.imagix3d_config import Imagix3DConfig
from autoencodix.data._global_vol_norm import GlobalVolumeNormalizer
from autoencodix.data._imgdataclass import ImgData
from autoencodix.data._xmodal_preprocessor import XModalPreprocessor
from autoencodix.data.datapackage import DataPackage
from autoencodix.utils._volreader import VolumeDataReader, VolumeNormalizer


class XModal3DPreprocessor(XModalPreprocessor):
    """XModalix preprocessor for 3D images.

    The class inherits from XModalPreprocessor and replaces the image-reading 
    and image-normalization steps by their 3D counterparts used by Imagix3D.
    """

    def __init__(
        self,
        config: XModalix3DConfig,
        ontologies: Optional[Union[Tuple, Dict]] = None,
    ):
        super().__init__(config=config, ontologies=ontologies)
        self.data_config = config.data_config
        self.config: XModalix3DConfig = config
        self.volume_scalers: Optional[Dict[str, GlobalVolumeNormalizer]] = None
        volume_config = cast(Imagix3DConfig, self.config)

        # Keep the normal XModalix multimodal processing, but replace the image
        # readers by the NIfTI/3D reader used in Imagix3D.
        self.data_readers[DataCase.IMG_TO_BULK]["img"] = VolumeDataReader(
            config=volume_config
        )
        self.data_readers[DataCase.SINGLE_CELL_TO_IMG]["img"] = VolumeDataReader(
            config=volume_config
        )
        self.data_readers[DataCase.IMG_TO_IMG] = VolumeDataReader(
            config=volume_config
        )

    def _normalize_image_data(
        self,
        images: List[ImgData],
        info_key: str,
    ) -> List[ImgData]:
        """Normalize 3D images using the Imagix3D volume normalization logic."""

        scaling_method = self.config.data_config.data_info[info_key].scaling
        if scaling_method == "NOTSET":
            scaling_method = self.config.scaling

        supported_methods = {"STANDARD", "MINMAX", "ROBUST", "NONE"}
        if scaling_method not in supported_methods:
            raise ValueError(
                f"Scaling method '{scaling_method}' is not supported for "
                f"3D image data. Supported methods are: "
                f"{sorted(supported_methods)}."
            )

        processed_images = []
        for img in images:
            img.img = VolumeNormalizer.normalize_volume(
                image=img.img,
                method=scaling_method,  # type: ignore[arg-type]
                normalize_nonzero_only=self.config.normalize_nonzero_only,
            )
            processed_images.append(img)

        return processed_images

    def _postsplit_volume_data(
        self,
        split_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Fit train-global volume scalers and apply them to all splits."""

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

                scaling_method = self.config.data_config.data_info[
                    modality_key
                ].scaling
                if scaling_method == "NOTSET":
                    scaling_method = self.config.scaling

                scaler = GlobalVolumeNormalizer.fit(
                    images=img_list,
                    method=scaling_method,  # type: ignore[arg-type]
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
        Keep the signature identical to BasePreprocessor.
        The actual 3D implementation lives in
        _process_img_to_img_case_volume().
        """
        out = self._process_img_to_img_case_volume(
            raw_user_data=raw_user_data
        )
        return cast(Dict[str, DataPackage], out)
    
    def _process_img_to_img_case_volume(
        self,
        raw_user_data: Optional[DataPackage] = None,
    ) -> Dict[str, Dict[str, Union[Any, DataPackage]]]:
        """Process an image-to-image XModalix case using 3D volumes."""

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
                key: self.filter_imgdata_list(img_list=value, ids=common_ids)
                for key, value in images.items()
            }

        if self.config.volume_scaling_strategy == "per_volume":

            def presplit_processor(
                modality_data: Dict[str, List[ImgData]],
            ) -> Dict[str, List[ImgData]]:
                return {
                    key: self._normalize_image_data(value, key)
                    for key, value in modality_data.items()
                }

            def postsplit_processor(
                split_data: Dict[str, Dict[str, Any]],
            ) -> Dict[str, Dict[str, Any]]:
                return split_data

        elif self.config.volume_scaling_strategy == "train_global":

            def presplit_processor(
                modality_data: Dict[str, List[ImgData]],
            ) -> Dict[str, List[ImgData]]:
                return modality_data

            def postsplit_processor(
                split_data: Dict[str, Dict[str, Any]],
            ) -> Dict[str, Dict[str, Any]]:
                return self._postsplit_volume_data(split_data=split_data)

        else:
            raise ValueError(
                "Unsupported volume_scaling_strategy: "
                f"{self.config.volume_scaling_strategy}"
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
