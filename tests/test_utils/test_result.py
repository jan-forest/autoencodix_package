from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from autoencodix.data._datasetcontainer import DataSetContainer
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils._result import Result
from autoencodix.utils._traindynamics import TrainingDynamics
from autoencodix.utils.default_config import DefaultConfig


# Unit Tests ------------------------------------------------------------------
class TestResultUnit:
    @pytest.fixture
    def mock_training_dynamics(self):
        mock = Mock(spec=TrainingDynamics)
        mock._data = {}
        return mock

    @pytest.fixture
    def empty_result(self, mock_training_dynamics):
        with patch(
            "autoencodix.utils._traindynamics.TrainingDynamics",
            return_value=mock_training_dynamics,
        ):
            return Result()

    def test_is_empty_value_empty_training_dynamics(self, empty_result):
        mock_td = Mock(spec=TrainingDynamics)
        mock_td._data = {}
        assert empty_result._is_empty_value(mock_td) is True

    @pytest.mark.parametrize(
        "data",
        [
            {0: {"train": np.array([])}},
            {0: {"train": {}}},
            {0: {"valid": np.array([])}},
            {0: {"test": np.array([])}},
        ],
    )
    def test_is_empty_value_nonempty_training_dynamics(self, empty_result, data):
        mock_td = Mock(spec=TrainingDynamics)
        mock_td._data = data
        assert empty_result._is_empty_value(mock_td) is False

    def test_is_empty_value_empty_tensor(self, empty_result):
        assert empty_result._is_empty_value(empty_result.preprocessed_data) is True

    def test_is_empty_value_nonempty_tensor(self, empty_result):
        assert empty_result._is_empty_value(torch.tensor([1, 2, 3])) is False

    def test_is_empty_value_empty_model(self, empty_result):
        model = Mock(spec=torch.nn.Module)
        model.parameters.return_value = []
        assert empty_result._is_empty_value(model) is True

    def test_is_empty_value_nonempty_model(self, empty_result):
        model = Mock(spec=torch.nn.Module)
        param = Mock()
        param.numel.return_value = 10
        model.parameters.return_value = [param]
        assert empty_result._is_empty_value(model) is False

    def test_is_empty_value_empty_dataset_container(self, empty_result):
        container = DataSetContainer(train=None, valid=None, test=None)
        assert empty_result._is_empty_value(container) is True

    def test_is_empty_value_nonempty_dataset_container(self, empty_result):
        mock_container = Mock(spec=DataSetContainer)
        mock_container.train = [1, 2, 3]
        mock_container.valid = None
        mock_container.test = None
        assert empty_result._is_empty_value(mock_container) is False

    def test_update_skips_empty_datasets(self, empty_result):
        mock_container = Mock(spec=DataSetContainer)
        mock_container.test = [1, 2, 3]
        empty_result.datasets = mock_container
        other = Result()
        empty_result.update(other)
        assert empty_result.datasets.test == [1, 2, 3]

    def test_update_skips_empty_model(self, empty_result):
        model = torch.nn.Linear(2, 2)
        empty_result.model = model
        other = Result()
        empty_result.update(other)
        assert empty_result.model is model

    def test_update_skips_empty_preprocessed_data(self, empty_result):
        data = torch.tensor([1, 2, 3])
        empty_result.preprocessed_data = data
        other = Result()
        empty_result.update(other)
        assert torch.equal(empty_result.preprocessed_data, data)

    def test_update_wrong_type_raises_error(self, empty_result):
        with pytest.raises(TypeError):
            empty_result.update("not a Result")


# Integration Tests -----------------------------------------------------------
class TestResultIntegration:
    @pytest.fixture
    def empty_result(self):
        return Result()

    @pytest.fixture
    def filled_result(self):
        result = Result()
        # Add training dynamics example (latentspaces only one since all TrainingDynamics work the same)
        result.latentspaces.add(epoch=0, split="train", data=np.array([1, 2, 3]))
        result.preprocessed_data = torch.tensor([4, 5, 6])
        result.model = torch.nn.Linear(2, 2)
        result.datasets = DataSetContainer(
            train=np.array([7, 8, 9]), valid=None, test=None
        )
        return result

    def test_class_init(self):
        result = Result()
        assert isinstance(result, Result)
        assert isinstance(result.latentspaces, TrainingDynamics)
        assert isinstance(result.reconstructions, TrainingDynamics)
        assert isinstance(result.mus, TrainingDynamics)
        assert isinstance(result.sigmas, TrainingDynamics)
        assert isinstance(result.losses, TrainingDynamics)
        assert isinstance(result.preprocessed_data, torch.Tensor)
        assert isinstance(result.model, torch.nn.Module)
        assert isinstance(result.model_checkpoints, TrainingDynamics)
        assert isinstance(result.datasets, DataSetContainer)

    def test_update_merges_training_dynamics(self, filled_result, empty_result):
        empty_result.update(filled_result)
        assert np.array_equal(empty_result.latentspaces, filled_result.latentspaces)

    def test_update_overwrites_tensor(self, filled_result, empty_result):
        empty_result.update(filled_result)
        assert torch.equal(
            empty_result.preprocessed_data, filled_result.preprocessed_data
        )

    def test_update_overwrites_model(self, empty_result, filled_result):
        model = torch.nn.Linear(2, 2)
        filled_result.model = model
        empty_result.update(filled_result)
        assert empty_result.model is model

    def test_update_overwrites_datasets(self, empty_result, filled_result):

        config = DefaultConfig()
        datasets = DataSetContainer(
            train=NumericDataset(torch.tensor([1, 2, 3]), config=config),
            valid=None,
            test=None,
        )
        filled_result.datasets = datasets
        empty_result.update(filled_result)
        assert empty_result.datasets is datasets

    def test_training_dynamics_accumulates_epochs(self, empty_result, filled_result):
        empty_result.latentspaces.add(epoch=0, split="train", data=np.array([1, 2, 3]))
        filled_result.latentspaces.add(epoch=1, split="train", data=np.array([4, 5, 6]))

        empty_result.update(filled_result)
        assert len(empty_result.latentspaces._data) == 2

    def test_update_with_mixed_data(self, empty_result, filled_result):
        # Add data to empty_result
        empty_result.latentspaces.add(epoch=0, split="train", data=np.array([1, 2, 3]))
        empty_result.preprocessed_data = torch.tensor([1, 2, 3])

        # Add different data to filled_result
        filled_result.latentspaces.add(epoch=1, split="train", data=np.array([4, 5, 6]))
        filled_result.preprocessed_data = torch.tensor([4, 5, 6])

        empty_result.update(filled_result)
        assert len(empty_result.latentspaces._data) == 2
        assert torch.equal(empty_result.preprocessed_data, torch.tensor([4, 5, 6]))

    def test_update_skips_empty_trainingdynamics(self, empty_result, filled_result):
        old_latentspaces = filled_result.latentspaces
        filled_result.update(empty_result)
        assert filled_result.latentspaces is old_latentspaces
