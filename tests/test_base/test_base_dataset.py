import pytest
import torch
from autoencodix.base._base_dataset import BaseDataset


class ConcreteDataset(BaseDataset):
    def __len__(self):
        return len(self.data)


class TestBaseDataset:
    @pytest.fixture
    def dummy_data(self):
        return torch.zeros(100, 10)

    @pytest.fixture
    def dummy_ids(self):
        return torch.arange(100)

    def test_get_input_dim(self, dummy_data):
        dataset = ConcreteDataset(dummy_data)
        assert dataset.get_input_dim() == 10
