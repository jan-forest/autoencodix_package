import numpy as np
import torch
import pytest
from autoencodix.utils._traindynamics import TrainingDynamics


# Unit Tests ------------------------------------------------------------------
# no integration tests necessary here
class TestTrainingDynamicsUnit:

    @pytest.fixture
    def dynamics(self):
        return TrainingDynamics()

    def test_initialization(self, dynamics):
        assert isinstance(dynamics, TrainingDynamics)

    def test_initial_data_empty(self, dynamics):
        assert dynamics._data == {}

    def test_add_epoch_data(self, dynamics):
        epoch = 0
        data = np.array([0.1, 0.2])
        split = "train"
        dynamics.add(epoch, data, split)
        assert np.array_equal(dynamics.get(epoch, split), data)

    def test_get_epoch_data(self, dynamics):
        epoch = 0
        data = np.array([0.1, 0.2])
        split = "train"
        dynamics.add(epoch, data, split)
        assert np.array_equal(dynamics.get(epoch)[split], data)

    def test_get_split_data(self, dynamics):
        epoch = 0
        data = np.array([0.1, 0.2])
        split = "train"
        dynamics.add(epoch, data, split)
        assert np.array_equal(dynamics.get(split=split), np.array([data]))

    def test_get_all_data(self, dynamics):
        epoch = 0
        data = np.array([0.1, 0.2])
        split = "train"
        dynamics.add(epoch, data, split)
        assert dynamics.get() == {epoch: {split: data}}

    def test_add_multiple_epochs_and_splits(self, dynamics):
        data1 = np.array([0.1, 0.2])
        data2 = np.array([0.2, 0.3])
        dynamics.add(data=data1, epoch=0, split="valid")
        dynamics.add(data=data2, epoch=1, split="valid")
        dynamics.add(data=data1, epoch=0, split="train")
        expected_data = {0: {"valid": data1, "train": data1}, 1: {"valid": data2}}
        assert np.array_equal(dynamics._data, expected_data)

    def test_get_data_multiple_epochs(self, dynamics):
        data1 = np.array([0.1, 0.2])
        data2 = np.array([0.2, 0.3])
        dynamics.add(0, data1, "train")
        dynamics.add(1, data2, "train")
        assert np.array_equal(dynamics.get(1, "train"), data2)

    def test_get_split_across_epochs(self, dynamics):
        data1 = np.array([0.1, 0.2])
        data2 = np.array([0.2, 0.3])
        dynamics.add(0, data1, "train")
        dynamics.add(1, data2, "train")
        assert np.array_equal(dynamics.get(split="train"), np.array([data1, data2]))

    def test_get_all_data_multiple_epochs(self, dynamics):
        data1 = np.array([0.1, 0.2])
        data2 = np.array([0.2, 0.3])
        dynamics.add(0, data1, "train")
        dynamics.add(1, data2, "train")
        assert dynamics.get() == {0: {"train": data1}, 1: {"train": data2}}

    def test_add_different_splits(self, dynamics):
        train_data = np.array([0.1, 0.2])
        valid_data = np.array([0.3, 0.4])
        dynamics.add(0, train_data, "train")
        dynamics.add(0, valid_data, "valid")
        assert np.array_equal(dynamics.get(0, "train"), train_data)

    def test_get_different_splits(self, dynamics):
        train_data = np.array([0.1, 0.2])
        valid_data = np.array([0.3, 0.4])
        dynamics.add(0, train_data, "train")
        dynamics.add(0, valid_data, "valid")
        assert np.array_equal(dynamics.get(0, "valid"), valid_data)

    def test_get_all_splits_for_epoch(self, dynamics):
        train_data = np.array([0.1, 0.2])
        valid_data = np.array([0.3, 0.4])
        dynamics.add(0, train_data, "train")
        dynamics.add(0, valid_data, "valid")
        dynamics.add(1, train_data, "train")  # should no be returned in get later
        assert dynamics.get(0) == {"train": train_data, "valid": valid_data}

    def test_get_all_data_different_splits(self, dynamics):
        train_data = np.array([0.1, 0.2])
        valid_data = np.array([0.3, 0.4])
        dynamics.add(0, train_data, "train")
        dynamics.add(0, valid_data, "valid")
        assert dynamics.get() == {0: {"train": train_data, "valid": valid_data}}

    def test_get_nonexistent_epoch(self, dynamics):
        dynamics.add(10, np.array([0.1, 0.2]), "train")
        assert dynamics.get(0) == {}

    def test_get_nonexistent_split(self, dynamics):
        dynamics.add(0, np.array([0.1, 0.2]), "test")
        assert np.array_equal(dynamics.get(0, "train"), np.array([]))

    def test_get_nonexistent_split_across_epochs(self, dynamics):
        dynamics.add(0, np.array([0.1, 0.2]), "valid")
        assert np.array_equal(dynamics.get(split="train"), np.array([]))

    def test_epochs(self, dynamics):
        dynamics.add(100, np.array([0.1, 0.2]), "train")
        dynamics.add(0, np.array([0.1, 0.2]), "train")
        dynamics.add(1, np.array([0.2, 0.3]), "train")
        assert dynamics.epochs() == [0, 1, 100]

    def test_getitem_epoch(self, dynamics):
        dynamics.add(0, np.array([0.1, 0.2]), "train")
        assert np.array_equal(dynamics[0]["train"], np.array([0.1, 0.2]))

    def test_getitem_slice(self, dynamics):
        dynamics.add(0, np.array([0.1, 0.2]), "train")
        dynamics.add(1, np.array([0.2, 0.3]), "train")
        dynamics.add(3, np.array([0.2, 0.3]), "train")
        dynamics.add(3, np.array([0.2, 0.3]), "valid")
        sliced = dynamics[0:2]
        train_first = sliced[0]["train"]
        train_second = sliced[1]["train"]
        assert np.array_equal(train_first, np.array([0.1, 0.2]))
        assert np.array_equal(train_second, np.array([0.2, 0.3]))

    @pytest.mark.parametrize(
        "invalid_data", [None, "string", torch.tensor([1, 2, 3]), [1, 2, 3]]
    )
    def test_invalid_add_data(self, dynamics, invalid_data):
        with pytest.raises(TypeError):
            dynamics.add(epoch=0, data=invalid_data, split="train")

    @pytest.mark.parametrize("data", [1.0, 0.5, 0.0, 3, 1000000, 0.0000001])
    def test_data_conversion(self, dynamics, data):
        dynamics.add(epoch=0, data=data, split="train") 
        retrieved_dynamics = dynamics.get(0, "train")
        print(retrieved_dynamics)
        # retrieved_data = retrieved_dynamics[0]["train"]
        assert retrieved_dynamics == data

    def test_add_invalid_split(self, dynamics):
        with pytest.raises(KeyError):
            dynamics.add(0, np.array([0.1, 0.2]), "invalid")

