import os
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from autoencodix.configs.default_config import DefaultConfig, DataInfo
from autoencodix.data._imgdataclass import ImgData


class ImageProcessingError(Exception):
    pass


class ImageSizeFinder:
    """Finds nearest quadratic image size that is dividable by 2^number_of_layers.ArithmeticError

    Nearest quadratic image size is based on the given image size in the config file.

    Attributes:
     config: Configuration object
     width: request image width by user.
     height: requested image height by user.

    """

    def __init__(self, config: DefaultConfig):
        """Inits the ImageSizeFinder

        Args:
            config: Configuration object.
        """
        self.config = config
        found_image_type = False
        for data_type in config.data_config.data_info.keys():
            print(f"Checking data type: {data_type}")
            if config.data_config.data_info[data_type].data_type == "IMG":
                print("Found image type in config")
                cur_data_info = config.data_config.data_info[data_type]
                print(f"current data info: {cur_data_info}")
                self.width = config.data_config.data_info[data_type].img_width_resize
                self.height = config.data_config.data_info[data_type].img_height_resize
                found_image_type = True
        if not found_image_type:
            raise ValueError("You need to provide a DATA_TYPE of with the TYPE key IMG")

        self.n_conv_layers = 5

    def _image_size_is_legal(self) -> bool:
        """Checks if image size is dividable by 2^number_of_layers to the givensize in the config.

        Returns:
            bool if image size is allowed.
        """
        return self.width % 2**self.n_conv_layers == 0

    def get_nearest_quadratic_image_size(self):
        """Finds nearest quadratic image size that is dividable by 2^number_of_layers

        Nearest quadratic image size is based on the given image size in the config file.

        Returns:
            Tuple of ints width and height with widht=height.
        Raies:
            ValueError: if not allowed image size can be found.
        """

        if self._image_size_is_legal():
            print(
                f"Given image size is possible, rescaling images to: {self.width}x{self.height}"
            )
            return self.width, self.height
        running_image_size = self.width
        while_loop_counter = 0
        while not running_image_size % 2**self.n_conv_layers == 0:
            running_image_size += 1
            while_loop_counter += 1
            if while_loop_counter > 10000:
                raise ValueError(
                    f"Could not find a quadratic image size that is dividable by 2^{self.n_conv_layers}"
                )
        print(
            f"Given image size{self.width}x{self.height} is not possible, rescaling to: {running_image_size}x{running_image_size}"
        )
        if running_image_size is None:
            raise ValueError(
                f"Could not find a quadratic image size that is dividable by 2^{self.n_conv_layers}"
            )

        return running_image_size, running_image_size


class ImageDataReader:
    """Reads and processes image data.

    Reads all images from the specified directory, processes them,
    and returns a list of ImgData objects.
    """

    def validate_image_path(self, image_path: Union[str, Path]) -> bool:
        """Checks if file extension is allowed:

        Allowed are (independent of capitalization):
            - jpg
            - jpeg
            - png
            - tif
            - tiff

        Args:
            image_path: path or str of image to read
        """
        path = Path(image_path) if isinstance(image_path, str) else image_path
        return (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )

    def parse_image_to_tensor(
        self,
        image_path: Union[str, Path],
        to_h: Optional[int] = None,
        to_w: Optional[int] = None,
    ) -> np.ndarray:
        """Reads an image from the given path, optionally resizes it, and converts it to a tensor.

        Args:
            image_path: The path to the image file.
            to_h: The desired height of the output tensor, by default None.
            to_w: The desired width of the output tensor, by default None.

        Returns:
            The processed image as a tensor.

        Raises:
            FileNotFoundError: If the image path is invalid or the image cannot be read.
            ImageProcessingError: If the image format is unsupported or an unexpected error occurs during processing.
        """

        if not self.validate_image_path(image_path):
            raise FileNotFoundError(f"Invalid image path: {image_path}")
        image_path = Path(image_path)
        SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageProcessingError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats are: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        try:
            if image_path.suffix.lower() in {".tif", ".tiff"}:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")

            (h, w, _) = image.shape[:3]
            if to_h is None:
                to_h = h
            if to_w is None:
                to_w = w

            if not (2 <= len(image.shape) <= 3):
                raise ImageProcessingError(
                    f"Image has unsupported shape: {image.shape}. "
                    "Supported shapes are 2D and 3D."
                )

            image = cv2.resize(image, (to_w, to_h), interpolation=cv2.INTER_AREA)

            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)

            image = image.transpose(2, 0, 1)
            return image

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ImageProcessingError, ValueError)):
                raise
            raise ImageProcessingError(
                f"Unexpected error during image processing: {str(e)}"
            )

    def read_all_images_from_dir(
        self,
        img_dir: str,
        to_h: Optional[int],
        to_w: Optional[int],
        annotation_df: pd.DataFrame,
        is_paired: Union[bool, None] = None,
    ) -> List[ImgData]:
        """Reads all images from a specified directory, processes them, returns list of ImgData objects.

        Args:
            img_dir: The directory containing the images.
            to_h: The desired height of the output tensors.
            to_w: The desired width of the output tensors.
            annotation_df: DataFrame containing image annotations.
            is_paired: Whether the images are paired with annotations.

        Returns:
            List of processed image data objects.

        Raises:
            ValueError: If the annotation DataFrame is missing required columns.
        """
        if "img_paths" not in annotation_df.columns:
            raise ValueError("img_paths column is missing in the annotation_df")

        SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if is_paired or is_paired is None:
            paths = [
                p
                for p in paths
                if os.path.basename(p) in annotation_df["img_paths"].tolist()
            ]
        imgs = []
        for p in paths:
            img = self.parse_image_to_tensor(image_path=p, to_h=to_h, to_w=to_w)
            img_path = os.path.basename(p)
            subset: Union[pd.Series, pd.DataFrame] = annotation_df[
                annotation_df["img_paths"] == img_path
            ]
            if not subset.empty:
                imgs.append(
                    ImgData(
                        img=img,
                        sample_id=subset["sample_ids"].iloc[0],
                        annotation=subset,
                    )
                )
        return imgs

    def read_annotation_file(self, data_info: DataInfo) -> pd.DataFrame:
        """Reads annotation file and returns DataFrame with file contents
        Args:
            data_info: specific part of the Configuration object for input data
        Returns:
            DataFrame with annotation data.

        """
        anno_file = (
            os.path.join(data_info.file_path)
            if data_info.extra_anno_file is None
            else os.path.join(data_info.extra_anno_file)
        )
        print(f"reading annotation file: {anno_file}")
        sep = data_info.sep
        if anno_file.endswith(".parquet"):
            annotation = pd.read_parquet(anno_file)
        elif anno_file.endswith((".csv", ".txt", ".tsv")):
            annotation = pd.read_csv(anno_file, sep=sep)
        else:
            raise ValueError(f"Unsupported file type for: {anno_file}")
        return annotation

    def read_data(
        self, config: DefaultConfig
    ) -> Tuple[Dict[str, List[ImgData]], Dict[str, pd.DataFrame]]:
        """Read image data from the specified directory based on configuration.

        Args:
            config: The configuration object containing the data configuration.

        Returns:
            A Tuple of Dicts:
            1. Dict with type of image data as key and actual List of ImgData as value.
            2. Dict with type of image data as key and DataFrame of annotation data as value.

        Raises:
            Exception: If no image data is found in the configuration or other validation errors occur.
        """
        # Find all image data sources in config
        image_sources = {
            k: v
            for k, v in config.data_config.data_info.items()
            if v.data_type == "IMG"
        }

        if not image_sources:
            raise ValueError("No image data found in the configuration.")

        result = {}
        annotation = {}
        for key, img_info in image_sources.items():
            try:
                result[key], annotation[key] = self._read_data(config, img_info)
                print(f"Successfully loaded {len(result[key])} images for {key}")
            except Exception as e:
                print(f"Error loading images for {key}: {str(e)}")
                # Decide whether to raise or continue based on your requirements

        return result, annotation

    def _read_data(
        self, config: DefaultConfig, img_info: DataInfo
    ) -> Tuple[List[ImgData], pd.DataFrame]:
        """Read data for a specific image source.

        Args:
            config: The configuration object containing the data configuration.
            img_info: The specific image info configuration.

        Returns:
            A Tuple of Dicts:
            1. Dict with type of image data as key and actual List of ImgData as value.
            2. Dict with type of image data as key and DataFrame of annotation data as value.

        """
        img_dir = img_info.file_path
        img_size_finder: ImageSizeFinder = ImageSizeFinder(config)
        to_h, to_w = img_size_finder.get_nearest_quadratic_image_size()

        if img_info.extra_anno_file is not None:
            # Use image-specific annotation file if provided
            annotation = self.read_annotation_file(img_info)
        else:
            # Otherwise use the global annotation file
            try:
                anno_info = next(
                    f
                    for f in config.data_config.data_info.values()
                    if f.data_type == "ANNOTATION"
                )
                if not config.requires_paired:
                    if config.requires_paired is not None:
                        raise ValueError(
                            "Img specific annotation file is required for unpaired translation."
                        )
                annotation = self.read_annotation_file(anno_info)
            except StopIteration:
                raise ValueError("No annotation data found in the configuration.")

        images = self.read_all_images_from_dir(
            img_dir=img_dir,
            to_h=to_h,
            to_w=to_w,
            annotation_df=annotation,
            is_paired=config.requires_paired,
        )
        annotations: pd.DataFrame = pd.concat([img.annotation for img in images])
        annotations.index = annotations.sample_ids

        del annotations["sample_ids"]
        return images, annotations


class ImageNormalizer:
    """Implements Normalization analog to other data modalites for ImageData"""

    @staticmethod
    def normalize_image(
        image: np.ndarray, method: Literal["STANDARD", "MINMAX", "ROBUST", "NONE"]
    ) -> np.ndarray:
        """Performs Image Normalization.

        Supported methods are:
            - STANDARD: (analog to StandardScaler of sklearn).
            - MINMAX: (analog to MinMaxSclaer of sklearn).
            - ROBUST: (analog to RobustScaler of sklearn).
            - NONE: no normalization.

        Args:
            image: input image as array.
            method: indicator string of which method to use
        Returns:
            The normalized images as np.ndarray
        Raises:
            ValueError: if unsupported normalization method is provided or Normalization fails for any other reason.
        """
        try:
            if method == "NONE":
                return image

            if method == "MINMAX":
                # Create a copy of the image for normalization
                normalized = image.astype(np.float32)
                cv2.normalize(
                    normalized,
                    normalized,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                )
                return normalized

            elif method == "STANDARD":
                mean = np.mean(image, axis=(1, 2), keepdims=True)
                std = np.std(image, axis=(1, 2), keepdims=True)
                return (image - mean) / (std + 1e-8)

            elif method == "ROBUST":
                median = np.median(image, axis=(1, 2), keepdims=True)
                q75, q25 = np.percentile(image, [75, 25], axis=(1, 2), keepdims=True)
                iqr = q75 - q25
                return (image - median) / (iqr + 1e-8)

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

        except Exception as e:
            raise ValueError(f"Failed to normalize image: {str(e)}")
