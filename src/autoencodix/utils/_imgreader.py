import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import cv2
import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from autoencodix.utils.default_config import DefaultConfig


@dataclass
class ImgData:
    img: np.ndarray
    sample_id: str
    annotation: pd.DataFrame


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

        if not ImageDataReader.validate_image_path(image_path):
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
        imgs = []
        if "img_path" not in annotation_df.columns:
            raise ValueError("img_path column is missing in the annotation_df")
        for p in paths:
            img = ImageDataReader.parse_image_to_tensor(
                image_path=p, to_h=to_h, to_w=to_w
            )
            img_path = os.path.basename(p)
            subset = annotation_df[annotation_df["img_path"] == img_path]
            imgs.append(
                ImgData(img=img, sample_id=subset["sample_ids"], annotation=subset)
            )
        return imgs

    def read_annotation_file(self, anno_info) -> pd.DataFrame:
        anno_file = Path(anno_info)
        sep = anno_info.sep
        if anno_file.endswith(".parquet"):
            annotation = pd.read_parquet(anno_file)
        elif anno_file.endswith((".csv", ".txt", ".tsv")):
            annotation = pd.read_csv(anno_file, sep=sep)
        else:
            raise ValueError(f"Unsupported file type for): {anno_file}")
        return annotation

    def read_data(self, config: DefaultConfig) -> List[ImgData]:
        print(f"reading data in imgreader")
        # get IMG datainfo
        img_info = next(
            f for f in config.data_config.data_info.values() if f.data_type == "IMG"
        )
        img_dir = img_info.file_path
        to_h = img_info.img_height_resize
        to_w = img_info.img_width_resize
        anno_info = next(
            f
            for f in config.data_config.data_info.values()
            if f.data_type == "ANNOTATION"
        )
        if img_info.extra_anno_file is not None:
            annotation = self.read_annotation_file(img_info.extra_anno_file)
        else:
            annotation = self.read_annotation_file(anno_info.file_path)
            raise ValueError(f"Unsupported file type for): {anno_info}")
        images = ImageDataReader.read_all_images_from_dir(
            img_dir=img_dir, to_h=to_h, to_w=to_w, annotation_df=annotation
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
