import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ImgData:
    """
    A data class for storing image data along with its associated metadata.

    Attributes:
        img (np.ndarray): The image data as a NumPy array.
        sample_id (str): A unique identifier for the image sample.
        annotation (pd.DataFrame): A DataFrame containing annotations or metadata related to the image.
    """
    img: np.ndarray
    sample_id: str
    annotation: pd.DataFrame

    def __repr__(self):
        return (f"ImgData(sample_id={self.sample_id!r}, "
                f"img_shape={self.img.shape}, "
                f"annotation_shape={self.annotation.shape})")
