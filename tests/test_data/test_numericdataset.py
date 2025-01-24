import pytest
import torch
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils.default_config import DefaultConfig


class TestNumericDataset:
    @pytest.fixture
    def default_config(self):
        return DefaultConfig()

    @pytest.fixture
    def dummy_data(self):
        data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = torch.tensor([0, 1, 2])
        return data, labels

    @pytest.fixture
    def dataset(self, dummy_data, default_config):
        data, labels = dummy_data
        return NumericDataset(data=data, config=default_config, ids=labels)

    def test_dataset_length(self, dataset, dummy_data):
        data, _ = dummy_data
        assert len(dataset.data) == len(
            data
        ), "Dataset length should match data length."

    def test_float_precision(self, dummy_data, default_config):
        data, labels = dummy_data
        default_config.float_precision = "float32"
        dataset = NumericDataset(data=data, config=default_config, ids=labels)
        assert (
            dataset.data.dtype == torch.float32
        ), "Data should be converted to float32 when provided as config parameter."
