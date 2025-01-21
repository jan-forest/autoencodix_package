import pytest
import torch
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils.default_config import DefaultConfig

# Sample Data for Testing
data = torch.tensor([[1, 2], [3, 4], [5, 6]])
labels = torch.tensor([0, 1, 0])
# mypy: ignore-errors
config = DefaultConfig() # type: ignore[call-arg]


@pytest.fixture
def default_config():
    return config


@pytest.fixture
def dummy_data():
    return data, labels


@pytest.fixture
def dataset(dummy_data):
    data, labels = dummy_data
    return NumericDataset(data=data, config=config, ids=labels)


def test_dataset_length(dataset):
    assert len(dataset) == len(data), "Dataset length should match data length."


def test_float_precision(dummy_data):
    data, labels = dummy_data
    config.float_precision = "float32"
    dataset = NumericDataset(data=data, config=config, ids=labels)
    assert (
        dataset.data.dtype == torch.float32
    ), "Data should be converted to float32 when provided as config parameter."
