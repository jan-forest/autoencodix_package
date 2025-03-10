import os
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict

import cv2
import numpy as np
import pandas as pd
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.data._imgdataclass import ImgData


class ImageProcessingError(Exception):
    pass


class ImageDataReader:
    """
    A utility class for reading and processing image data.

    Methods
    -------
    validate_image_path(image_path: Union[str, Path]) -> bool
        Validates if the given image path exists, is a file, and has a supported image extension.

    parse_image_to_tensor(image_path: Union[str, Path], to_h: Optional[int] = None, to_w: Optional[int] = None) -> np.ndarray
        Reads an image from the given path, optionally resizes it, and converts it to a tensor.

    read_all_images_from_dir(img_dir: str, to_h: Optional[int], to_w: Optional[int], annotation_df: Optional[pd.DataFrame]) -> List[ImgData]
        Reads all images from the specified directory, processes them, and returns a list of ImgData objects.
    """

    def __init__(self):
        pass

    def validate_image_path(self, image_path: Union[str, Path]) -> bool:
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
        """
        Reads an image from the given path, optionally resizes it, and converts it to a tensor.

        Parameters
        ----------
        image_path : Union[str, Path]
            The path to the image file.
        to_h : Optional[int], optional
            The desired height of the output tensor, by default None.
        to_w : Optional[int], optional
            The desired width of the output tensor, by default None.

        Returns
        -------
        np.ndarray
            The processed image as a tensor.

        Raises
        ------
        FileNotFoundError
            If the image path is invalid or the image cannot be read.
        ImageProcessingError
            If the image format is unsupported or an unexpected error occurs during processing.
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
        annotation_df: Optional[pd.DataFrame],
        is_paired: Union[bool, None] = None,
    ) -> List[ImgData]:
        """
        Reads all images from the specified directory, processes them, and returns a list of ImgData objects.

        Parameters
        ----------
        img_dir : str
            The directory containing the images.
        to_h : Optional[int], optional
            The desired height of the output tensors, by default Non
            If the annotation DataFrame is missing required columns.
        """

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
        if "img_paths" not in annotation_df.columns:
            raise ValueError("img_paths column is missing in the annotation_df")
        for p in paths:
            img = self.parse_image_to_tensor(image_path=p, to_h=to_h, to_w=to_w)
            img_path = os.path.basename(p)
            subset = annotation_df[annotation_df["img_paths"] == img_path]
            imgs.append(
                ImgData(img=img, sample_id=subset["sample_ids"], annotation=subset)
            )
        return imgs

    def read_annotation_file(self, data_info) -> pd.DataFrame:
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

    def read_data(self, config: DefaultConfig) -> Dict[str, List[ImgData]]:
        """
        Read image data from the specified directory based on configuration.

        Parameters
        ----------
        config : DefaultConfig
            The configuration object containing the data configuration.

        Returns
        -------
        Dict[str, List[ImgData]]
            A dictionary mapping data keys to lists of ImgData objects.
            If only one image source is found, returns a list of ImgData objects directly.

        Raises
        ------
        ValueError
            If no image data is found in the configuration or other validation errors occur.
        """
        # Find all image data sources in config
        image_sources = {
            k: v
            for k, v in config.data_config.data_info.items()
            if v.data_type == "IMG"
        }

        if not image_sources:
            raise ValueError("No image data found in the configuration.")

        # If only one image source, return the data directly
        if len(image_sources.keys()) == 1:
            key = next(iter(image_sources.keys()))
            return {key: self._read_data(config=config, img_info=image_sources[key])}

        # Otherwise, process each image source
        result = {}
        for key, img_info in image_sources.items():
            try:
                result[key] = self._read_data(config, img_info)
                print(f"Successfully loaded {len(result[key])} images for {key}")
            except Exception as e:
                print(f"Error loading images for {key}: {str(e)}")
                # Decide whether to raise or continue based on your requirements

        return result

    def _read_data(self, config: DefaultConfig, img_info) -> List[ImgData]:
        """
        Read data for a specific image source.

        Parameters
        ----------
        config : DefaultConfig
            The configuration object containing the data configuration.
        img_info :
            The specific image info configuration.

        Returns
        -------
        List[ImgData]
            List of ImgData objects for this image source.
        """
        img_dir = img_info.file_path
        to_h = img_info.img_height_resize
        to_w = img_info.img_width_resize

        # Get annotation data
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
                if not config.paired_translation:
                    if config.paired_translation is not None:
                        
                        raise ValueError(
                            "Img specific annotation file is required for unpaired translation."
                    )
                annotation = self.read_annotation_file(anno_info)
            except StopIteration:
                raise ValueError("No annotation data found in the configuration.")

        # Read and process the images
        images = self.read_all_images_from_dir(
            img_dir=img_dir,
            to_h=to_h,
            to_w=to_w,
            annotation_df=annotation,
            is_paired=config.paired_translation,
        )

        return images


class ImageNormalizer:
    @staticmethod
    def normalize_image(
        image: np.ndarray, method: Literal["STANDARD", "MINMAX", "ROBUST", "NONE"]
    ) -> np.ndarray:
        try:
            if method == "NONE":
                return image

            if method == "MINMAX":
                return cv2.normalize(
                    image,
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                )

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
