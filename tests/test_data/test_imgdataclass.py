# test_img_data.py

import numpy as np
import pandas as pd
import pytest
from autoencodix.data._imgdataclass import ImgData


@pytest.fixture
def example_img_data():
    img = np.zeros((64, 64, 3))
    sample_id = "sample_001"
    annotation = pd.DataFrame({"label": ["cat"], "confidence": [0.95]})
    return ImgData(img=img, sample_id=sample_id, annotation=annotation)


def test_imgdata_attributes(example_img_data):
    assert example_img_data.img.shape == (64, 64, 3)


def test_imgdata_sample_id(example_img_data):
    assert example_img_data.sample_id == "sample_001"


def test_imgdata_annotation(example_img_data):
    assert isinstance(example_img_data.annotation, pd.DataFrame)


def test_imgdata_repr(example_img_data):
    repr_str = repr(example_img_data)
    assert "sample_id='sample_001'" in repr_str
