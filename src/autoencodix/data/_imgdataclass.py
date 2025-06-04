import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ImgData:
    """Stores image data along with its associated metadata.

    Attributes:
        img : The image data as a NumPy array.
        sample_id: A unique identifier for the image sample.
        annotation: A DataFrame containing annotations or metadata related to the image.
    """

    img: np.ndarray
    sample_id: str
    annotation: pd.DataFrame

    def __repr__(self):
        return (
            f"ImgData(\n"
            f"    sample_id={self.sample_id!r},\n"
            f"    img_shape={self.img.shape},\n"
            f"    annotation_shape={self.annotation.shape}\n"
            f")"
        )
