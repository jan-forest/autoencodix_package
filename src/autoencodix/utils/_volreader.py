import os
from pathlib import Path
from typing import List, Literal, Union, Dict, Tuple

import numpy as np
import pandas as pd
import torchio as tio
from autoencodix.configs import Imagix3DConfig, DataInfo
from autoencodix.data._imgdataclass import ImgData

SUPPORTED_EXTENSIONS = (".nii", ".nii.gz")  # for now

class ImageProcessingError(Exception):
    pass


class VolumeSizeFinder:
    """Finds 3D image dimensions that are dividable by 2^number_of_layers

    Nearest quadratic image size is based on the given image size in the config file.

    Attributes:
     config: Configuration object
     dim_to_multiple: The size by which the side of a dimension must be a multiple
     vol_shape: the exact spatial dimensions that the 3D image should have
     shape_policy: how to pad and/or crop 3D images

    """

    def __init__(self, config: Imagix3DConfig, dims: Tuple[int, int, int]):
        """Inits the ImageSizeFinder

        Args:
            config: Configuration object.
        """
        self.config = config
        found_image_type = False
        for data_type in config.data_config.data_info.keys():
            if config.data_config.data_info[data_type].data_type == "IMG":
                self.dim_to_multiple = config.target_multiple
                self.vol_shape = config.target_shape_3d
                self.shape_policy = config.spatial_shape_policy
                self.dim_1, self.dim_2, self.dim_3 = dims
                found_image_type = True
        if not found_image_type:
            raise ValueError("You need to provide a DATA_TYPE of with the TYPE key IMG")
        
        self.n_conv_layers = config.n_conv_layers_3d


    def get_volume_image_dimensions(self) -> Tuple[int, int, int]: 
        """Finds nearest quadratic image size that is dividable by 2^number_of_layers

        Nearest quadratic image size is based on the given image size in the config file.

        Returns:
            Tuple of ints width and height with widht=height.
        Raies:
            ValueError: if not allowed image size can be found.
        """
       
        mult = self.dim_to_multiple
        dist2target_1 = (self.dim_1 % mult) #type: ignore
        dist2target_2 = (self.dim_2 % mult) #type: ignore
        dist2target_3 = (self.dim_3 % mult) #type: ignore
        
        if self.shape_policy == "pad_to_multiple":
            self.dim_1 += (mult - dist2target_1) % mult #type: ignore
            self.dim_2 += (mult - dist2target_2) % mult #type: ignore
            self.dim_3 += (mult - dist2target_3) % mult #type: ignore
            
        elif self.shape_policy == "crop_to_multiple":
            self.dim_1 -= dist2target_1 #type: ignore
            self.dim_2 -= dist2target_2 #type: ignore
            self.dim_3 -= dist2target_3 #type: ignore
        
        elif self.shape_policy == "crop_or_pad_to_multiple":
            if mult - dist2target_1 < dist2target_1: #type: ignore
                self.dim_1 += mult - dist2target_1 #type: ignore
            else:
                self.dim_1 -= dist2target_1 #type: ignore
                
            if mult - dist2target_2 < dist2target_2: #type: ignore
                self.dim_2 += mult - dist2target_2 #type: ignore
            else:
                self.dim_2 -= dist2target_2 #type: ignore
                
            if mult - dist2target_3 < dist2target_3: #type: ignore
                self.dim_3 += mult - dist2target_3 #type: ignore
            else:
                self.dim_3 -= dist2target_3 #type: ignore
        
        elif self.shape_policy == "crop_or_pad_to_shape":
            self.dim_1, self.dim_2, self.dim_3 = self.vol_shape #type: ignore

        return self.dim_1, self.dim_2, self.dim_3


class VolumeDataReader:
    """Reads and processes image data.

    Reads all 3D images (volumes) from the specified directory, processes them,
    and returns a list of ImgData objects.
    """

    def __init__(self, config: Imagix3DConfig):
        self.config = config
        self.shape_policy = config.spatial_shape_policy

    def validate_image_path(self, image_path: Union[str, Path]) -> bool:
        """Checks if file extension is allowed:

        Allowed are NIfTI files (independent of capitalization):
            - nii
            - nii.gz

        Args:
            image_path: path or str of image to read
        """
        path = Path(image_path) if isinstance(image_path, str) else image_path
        return (
            path.exists()
            and path.is_file()
            and path.name.lower().endswith(SUPPORTED_EXTENSIONS)
        )

    def parse_image_to_array( # parse_image_to_tensor
        self,
        image_path: Union[str, Path]
    ) -> np.ndarray:
        """Reads an image from the given path, optionally crops or pads it, and converts it to a numpy array.

        Args:
            image_path: The path to the image file.

        Returns:
            The processed image as a tensor.

        Raises:
            FileNotFoundError: If the image path is invalid or the image cannot be read.
            ImageProcessingError: If the image format is unsupported or an unexpected error occurs during processing.
        """

        if not self.validate_image_path(image_path):
            raise FileNotFoundError(f"Invalid image path: {image_path}")
        image_path = Path(image_path)
        if not image_path.name.lower().endswith(SUPPORTED_EXTENSIONS):
            raise ImageProcessingError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats are: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        try:
            # load image from path
            image = tio.ScalarImage(image_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
        
            # ensures that 3D images (like MRI or CT scans) are reoriented to a RAI 
            # (Right-to-Left, Anterior-to-Posterior, Inferior-to-Superior) orientation
            image = tio.transforms.ToCanonical()(image)

            # get underlying tensor
            tensor = image.data  # Torch tensor
            dims = tuple(tensor.shape[1:])  # spatial dims only
            
            img_size_finder: VolumeSizeFinder = VolumeSizeFinder(self.config, dims) #type: ignore
            to_d1, to_d2, to_d3 = img_size_finder.get_volume_image_dimensions()
            padding_mode = self.config.padding_mode_3d
            
            image = tio.transforms.CropOrPad((to_d1, to_d2, to_d3), padding_mode=padding_mode)(image)
            tensor = image.data
            image = tensor.numpy()
            image = image.transpose(0, 3, 2, 1) # from (C, W, H, D) to (C, D, H, W)

            return image

        except Exception as e:
            raise e

    def read_all_images_from_dir(
        self,
        img_dir: str,
        annotation_df: pd.DataFrame,
        is_paired: Union[bool, None] = None,
    ) -> List[ImgData]:
        """Reads all images from a specified directory, processes them, returns list of ImgData objects.

        Args:
            img_dir: The directory containing the images.
            annotation_df: DataFrame containing image annotations.
            is_paired: Whether the images are paired with annotations.

        Returns:
            List of processed image data objects.

        Raises:
            ValueError: If the annotation DataFrame is missing required columns.
        """
        if self.config.img_path_col not in annotation_df.columns:
            raise ValueError(
                f" The defined column for image paths: {self.config.img_path_col} column is missing in the annotation_df\
                             you can define this in the config via the param `img_path_col`"
            )

        paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if Path(f).name.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        if is_paired or is_paired is None:
            paths = [
                p
                for p in paths
                if os.path.basename(p)
                in annotation_df[self.config.img_path_col].tolist()
            ]
        imgs = []
        for p in paths:
            img = self.parse_image_to_array(image_path=p)
            img_path = os.path.basename(p)
            subset: Union[pd.Series, pd.DataFrame] = annotation_df[
                annotation_df[self.config.img_path_col] == img_path
            ]
            if not subset.empty:
                imgs.append(
                    ImgData(
                        img=img,
                        sample_id=str(subset.index[0]),
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
        sep = data_info.sep
        if anno_file.endswith(".parquet"):
            annotation = pd.read_parquet(anno_file)
        elif anno_file.endswith((".csv", ".txt", ".tsv")):
            annotation = pd.read_csv(anno_file, sep=sep, index_col=0, engine="python")
        else:
            raise ValueError(f"Unsupported file type for: {anno_file}")
        return annotation

    def read_data(
        self, config: Imagix3DConfig
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
        self, config: Imagix3DConfig, img_info: DataInfo
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

        if img_info.extra_anno_file is not None:
            # Use image-specific annotation file if provided
            annotation = self.read_annotation_file(img_info)
        else:
            # Otherwise use the global annotation file
            try:
                anno_info = next(
                    f for f in config.data_config.data_info.values()
                    if f.data_type == "ANNOTATION"
                )
                annotation = self.read_annotation_file(anno_info)
            except StopIteration:
                raise ValueError("No annotation data found in the configuration.")

        images = self.read_all_images_from_dir(
            img_dir=img_dir,
            annotation_df=annotation,
            is_paired=config.requires_paired,
        )
        annotations: pd.DataFrame = pd.concat([img.annotation for img in images])

        return images, annotations


class VolumeNormalizer:
    """Implements Normalization analog to other data modalites for ImageData"""

    @staticmethod
    def normalize_volume(
        image: np.ndarray, 
        method: Literal["STANDARD", "MINMAX", "ROBUST", "NONE"],
        normalize_nonzero_only: bool
    ) -> np.ndarray:
        """Performs 3D Image (Volume) Normalization.

        Supported methods are:
            - STANDARD: (analog to StandardScaler of sklearn).
            - MINMAX: (analog to MinMaxSclaer of sklearn).
            - ROBUST: (analog to RobustScaler of sklearn).
            - NONE: no normalization.

        Args:
            image: input image as array.
            method: indicator string of which method to use
            normalize_nonzero_only: if True, compute normalization statistics
                only on nonzero voxels and leave zero voxels unchanged.
            
        Returns:
            The normalized images as np.ndarray
        Raises:
            ValueError: if unsupported normalization method is provided or Normalization fails for any other reason.
        """
        try:
            if method == "NONE":
                return image
            
            image = image.astype(np.float32, copy=True)
            eps = np.finfo(np.float32).eps # machine epsilon for that data type
            
            # Only non-zero voxels will get scaled
            if normalize_nonzero_only:
                for c in range(image.shape[0]): # relevant in case there is more than 1 channel
                    channel = image[c]
                    mask = channel != 0

                    # no nonzero voxels -> leave channel unchanged
                    if not np.any(mask):
                        continue

                    vals = channel[mask]

                    if method == "MINMAX":
                        vmin = np.min(vals)
                        vmax = np.max(vals)
                        channel[mask] = (vals - vmin) / ((vmax - vmin) + eps)

                    elif method == "STANDARD":
                        mean = np.mean(vals)
                        std = np.std(vals)
                        channel[mask] = (vals - mean) / (std + eps)

                    elif method == "ROBUST":
                        median = np.median(vals)
                        q75, q25 = np.percentile(vals, [75, 25])
                        iqr = q75 - q25
                        channel[mask] = (vals - median) / (iqr + eps)
                    
                    else:
                        raise ValueError(f"Unsupported normalization method: {method}")
                    
                    image[c] = channel
                    
                return image
            
            # Scaling is applied to all voxels (zero-value voxels included)   
            if method == "MINMAX":
                minimum = np.min(image, axis=(1, 2, 3), keepdims=True)
                maximum = np.max(image, axis=(1, 2, 3), keepdims=True)
                return (image - minimum) / ((maximum - minimum) + eps)
           
            elif method == "STANDARD":
                mean = np.mean(image, axis=(1, 2, 3), keepdims=True)
                std = np.std(image, axis=(1, 2, 3), keepdims=True)
                return (image - mean) / (std + eps)

            elif method == "ROBUST":
                median = np.median(image, axis=(1, 2, 3), keepdims=True)
                q75, q25 = np.percentile(image, [75, 25], axis=(1, 2, 3), keepdims=True)
                iqr = q75 - q25
                return (image - median) / (iqr + eps)

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

        except Exception as e:
            raise ValueError(f"Failed to normalize image: {str(e)}")
