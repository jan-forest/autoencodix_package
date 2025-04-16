import pytest
import torch

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.data._datasetcontainer import DatasetContainer


# makes sure to separate the DatasetContainer tests from the BaseDataset tests
# we use this for unit tests and for integration tests we use the real BaseDataset
class MockBaseDataset:
    pass


class TestUnitDatasetContainer:
    @pytest.fixture
    def empty_container(self):
        """Provides an empty DatasetContainer for testing initialization and gradual population."""
        return DatasetContainer()

    @pytest.fixture
    def mock_datasets(self):
        """Provides a set of mock datasets with different characteristics."""
        return {
            "train": MockBaseDataset(),
            "valid": MockBaseDataset(),
            "test": MockBaseDataset(),
        }

    @pytest.fixture
    def filled_container(self, mock_datasets):
        """Provides a fully populated container to test operations on complete sets."""
        return DatasetContainer(
            train=mock_datasets["train"],
            valid=mock_datasets["valid"],
            test=mock_datasets["test"],
        )

    def test_dataset_container_initialization_train(self, empty_container):
        """Verify that new containers initialize with all slots as None."""
        assert empty_container.train is None

    def test_dataset_container_initialization_valid(self, empty_container):
        """Verify that new containers initialize with all slots as None."""
        assert empty_container.valid is None

    def test_dataset_container_initialization_test(self, empty_container):
        """Verify that new containers initialize with all slots as None."""
        assert empty_container.test is None

    @pytest.mark.parametrize("key", ["tran", "vaild", "tset"])
    def test_invalid_key_getter(self, filled_container, key):
        """Test assignment of invalid types to ensure type checking."""
        with pytest.raises(KeyError):
            _ = filled_container[key]

    @pytest.mark.parametrize("key", ["tran", "vaild", "tset"])
    def test_invalid_key_setter(self, filled_container, mock_datasets, key):
        """Test assignment of invalid types to ensure type checking."""
        with pytest.raises(KeyError):
            filled_container[key] = mock_datasets["train"]

    def test_partial_initialization(self):
        """Test container behavior with only some slots populated."""
        container = DatasetContainer(train=MockBaseDataset())
        assert isinstance(container.train, MockBaseDataset)
        assert container.valid is None
        assert container.test is None

    def test_key_error_messages_access(self, empty_container, mock_datasets):
        """Verify that appropriate error messages are raised for invalid operations."""
        # Invalid key access
        with pytest.raises(KeyError) as exc_info:
            _ = empty_container["invalid"]
        assert "Must be 'train', 'valid', or 'test'" in str(exc_info.value)

    def test_key_error_messages_assignment(self, empty_container, mock_datasets):
        # Invalid key assignment
        with pytest.raises(KeyError) as exc_info:
            empty_container["invalid"] = mock_datasets["train"]
        assert "Must be 'train', 'valid', or 'test'" in str(exc_info.value)


# Integration Tests with Real BaseDataset
class TestIntegrationDatasetContainer:
    @pytest.fixture
    def real_dataset(self):
        """Provides a real BaseDataset instance for integration testing."""
        # Configure with minimum required parameters
        return BaseDataset(data=torch.tensor([[1, 2], [3, 4], [5, 6]]))

    @pytest.fixture
    def real_filled_container(self, real_dataset):
        """Provides a fully populated container to test operations on complete sets."""
        return DatasetContainer(
            train=real_dataset, valid=real_dataset, test=real_dataset
        )

    def test_integration_with_real_dataset(self, real_filled_container, real_dataset):
        """Verify compatibility with actual BaseDataset instances."""
        assert isinstance(real_filled_container.train, BaseDataset)
        assert isinstance(real_filled_container["valid"], BaseDataset)

        if hasattr(real_dataset, "data"):
            assert real_filled_container.train.data.shape == real_dataset.data.shape
        if hasattr(real_dataset, "ids"):
            assert real_filled_container.train.sample_ids is real_dataset.ids
