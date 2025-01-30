from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils._result import Result, LossRegistry
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
        container = DatasetContainer(train=None, valid=None, test=None)
        assert empty_result._is_empty_value(container) is True

    def test_is_empty_value_nonempty_dataset_container(self, empty_result):
        mock_container = Mock(spec=DatasetContainer)
        mock_container.train = [1, 2, 3]
        mock_container.valid = None
        mock_container.test = None
        assert empty_result._is_empty_value(mock_container) is False

    def test_update_skips_empty_datasets(self, empty_result):
        mock_container = Mock(spec=DatasetContainer)
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
        result.datasets = DatasetContainer(
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
        assert isinstance(result.datasets, DatasetContainer)

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
        datasets = DatasetContainer(
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

    def test_complex_loss_update(self):
        to_update = {
            0: {"train": 0.0, "valid": 0.0, "test": 0.0},  # totally overwrite
            2: {"train": 0.0, "valid": 0.0, "test": 0.0},  # keep all
            3: {},  # update with partial data
            4: None,  # udate with all
            5: {
                "train": None,
                "valid": 0.0,
                "test": 0.0,
            },  # update with partial data (keep test)
            6: {
                "train": 0.0
            },  # update with partial data (keep train), add vlaid and test
            7: {"train": 0},  # update with partial data {train: 0.1}
            100: {"train": 0.0, "valid": 0.0, "test": 0.0},  # not in to update
        }

        update_with = {
            0: {"train": 0.1, "valid": 0.2, "test": 0.3},
            1: {"train": 0.4, "valid": 0.5, "test": 0.6},
            2: {},
            3: {"train": 0.7, "valid": 0.8},
            4: {"train": 0.9, "valid": 1.0, "test": 1.1},
            5: {"train": 0.12, "valid": 0.22},
            6: {"valid": 0.32, "test": 0.42},
            7: {"train": 0.1},
        }

        expected_result = {
            0: {"train": np.array(0.1), "valid": np.array(0.2), "test": np.array(0.3)},
            1: {"train": np.array(0.4), "valid": np.array(0.5), "test": np.array(0.6)},
            2: {"train": np.array(0.0), "valid": np.array(0.0), "test": np.array(0.0)},
            3: {"train": np.array(0.7), "valid": np.array(0.8)},
            4: {"train": np.array(0.9), "valid": np.array(1.0), "test": np.array(1.1)},
            5: {
                "train": np.array(0.12),
                "valid": np.array(0.22),
                "test": np.array(0.0),
            },
            6: {
                "train": np.array(0.0),
                "valid": np.array(0.32),
                "test": np.array(0.42),
            },
            7: {"train": np.array(0.1)},
            100: {
                "train": np.array(0.0),
                "valid": np.array(0.0),
                "test": np.array(0.0),
            },
        }

        to_update_dyn = TrainingDynamics(_data=to_update)
        to_update_registry = LossRegistry(_losses={"recon_loss": to_update_dyn})
        to_update_result = Result(sub_losses=to_update_registry)

        update_with_dyn = TrainingDynamics(_data=update_with)
        update_with_registry = LossRegistry(_losses={"recon_loss": update_with_dyn})
        update_with_result = Result(sub_losses=update_with_registry)
        to_update_result.update(update_with_result)
        after_update = dict(
            sorted(to_update_result.sub_losses.get(key="recon_loss").get().items())
        )
        assert after_update == expected_result

    @pytest.mark.parametrize(
        "to_update, update_with, expected_result",
        [
            (
                {
                    0: {"train": 0.0, "valid": 0.0, "test": 0.0},
                    2: {"train": 0.0, "valid": 0.0, "test": 0.0},
                    3: {},
                    4: None,
                    5: {"train": None, "valid": 0.0, "test": 0.0},
                    6: {"train": 0.0},
                    7: {"train": 0},
                    100: {"train": 0.0, "valid": 0.0, "test": 0.0},
                },
                {
                    0: {"train": 0.1, "valid": 0.2, "test": 0.3},
                    1: {"train": 0.4, "valid": 0.5, "test": 0.6},
                    2: {},
                    3: {"train": 0.7, "valid": 0.8},
                    4: {"train": 0.9, "valid": 1.0, "test": 1.1},
                    5: {"train": 0.12, "valid": 0.22},
                    6: {"valid": 0.32, "test": 0.42},
                    7: {"train": 0.1},
                },
                {
                    0: {
                        "train": np.array(0.1),
                        "valid": np.array(0.2),
                        "test": np.array(0.3),
                    },
                    1: {
                        "train": np.array(0.4),
                        "valid": np.array(0.5),
                        "test": np.array(0.6),
                    },
                    2: {
                        "train": np.array(0.0),
                        "valid": np.array(0.0),
                        "test": np.array(0.0),
                    },
                    3: {"train": np.array(0.7), "valid": np.array(0.8)},
                    4: {
                        "train": np.array(0.9),
                        "valid": np.array(1.0),
                        "test": np.array(1.1),
                    },
                    5: {
                        "train": np.array(0.12),
                        "valid": np.array(0.22),
                        "test": np.array(0.0),
                    },
                    6: {
                        "train": np.array(0.0),
                        "valid": np.array(0.32),
                        "test": np.array(0.42),
                    },
                    7: {"train": np.array(0.1)},
                    100: {
                        "train": np.array(0.0),
                        "valid": np.array(0.0),
                        "test": np.array(0.0),
                    },
                },
            ),
            (
                {
                    0: {"train": 0.0, "valid": 0.0, "test": 0.0},
                    1: {"train": 0.0, "valid": 0.0, "test": 0.0},
                },
                {
                    0: {"train": 0.1, "valid": 0.2, "test": 0.3},
                    1: {"train": 0.4, "valid": 0.5, "test": 0.6},
                },
                {
                    0: {
                        "train": np.array(0.1),
                        "valid": np.array(0.2),
                        "test": np.array(0.3),
                    },
                    1: {
                        "train": np.array(0.4),
                        "valid": np.array(0.5),
                        "test": np.array(0.6),
                    },
                },
            ),
            (
                {
                    0: {"train": 0.0, "valid": 0.0, "test": 0.0},
                },
                {
                    0: {"train": 0.1, "valid": 0.2, "test": 0.3},
                    1: {"train": 0.4, "valid": 0.5, "test": 0.6},
                },
                {
                    0: {
                        "train": np.array(0.1),
                        "valid": np.array(0.2),
                        "test": np.array(0.3),
                    },
                    1: {
                        "train": np.array(0.4),
                        "valid": np.array(0.5),
                        "test": np.array(0.6),
                    },
                },
            ),
        ],
    )
    def test_update_with_various_losses(self, to_update, update_with, expected_result):
        to_update_dyn = TrainingDynamics(_data=to_update)
        to_update_registry = LossRegistry(_losses={"recon_loss": to_update_dyn})
        to_update_result = Result(sub_losses=to_update_registry)

        update_with_dyn = TrainingDynamics(_data=update_with)
        update_with_registry = LossRegistry(_losses={"recon_loss": update_with_dyn})
        update_with_result = Result(sub_losses=update_with_registry)

        to_update_result.update(update_with_result)
        after_update = dict(
            sorted(to_update_result.sub_losses.get(key="recon_loss").get().items())
        )
        assert after_update == expected_result

    def test_update_with_multiple_losses(self):
        to_update = {
            0: {"train": np.array(0.0), "valid": np.array(0.0), "test": np.array(0.0)},
            2: {"train": np.array(0.0), "valid": np.array(0.0), "test": np.array(0.0)},
            3: {},
            4: None,
            5: {"train": None, "valid": np.array(0.0), "test": np.array(0.0)},
            6: {"train": np.array(0.0)},
            7: {"train": np.array(0)},
            100: {
                "train": np.array(0.0),
                "valid": np.array(0.0),
                "test": np.array(0.0),
            },
        }

        update_with = {
            0: {"train": 0.1, "valid": 0.2, "test": 0.3},
            1: {"train": 0.4, "valid": 0.5, "test": 0.6},
            2: {},
            3: {"train": 0.7, "valid": 0.8},
            4: {"train": 0.9, "valid": 1.0, "test": 1.1},
            5: {"train": 0.12, "valid": 0.22},
            6: {"valid": 0.32, "test": 0.42},
            7: {"train": 0.1},
        }

        expected_result = {
            0: {"train": np.array(0.1), "valid": np.array(0.2), "test": np.array(0.3)},
            1: {"train": np.array(0.4), "valid": np.array(0.5), "test": np.array(0.6)},
            2: {"train": np.array(0.0), "valid": np.array(0.0), "test": np.array(0.0)},
            3: {"train": np.array(0.7), "valid": np.array(0.8)},
            4: {"train": np.array(0.9), "valid": np.array(1.0), "test": np.array(1.1)},
            5: {
                "train": np.array(0.12),
                "valid": np.array(0.22),
                "test": np.array(0.0),
            },
            6: {
                "train": np.array(0.0),
                "valid": np.array(0.32),
                "test": np.array(0.42),
            },
            7: {"train": np.array(0.1)},
            100: {
                "train": np.array(0.0),
                "valid": np.array(0.0),
                "test": np.array(0.0),
            },
        }

        to_update_dyn = TrainingDynamics(_data=to_update)
        to_update_registry = LossRegistry(
            _losses={
                "recon_loss": to_update_dyn,
                "var_loss": to_update_dyn,
                "more_loss": TrainingDynamics(_data={}),
            }
        )
        to_update_result = Result(sub_losses=to_update_registry)

        update_with_dyn = TrainingDynamics(_data=update_with)
        update_with_registry = LossRegistry(
            _losses={
                "recon_loss": update_with_dyn,
                "var_loss": TrainingDynamics(_data={}),
                "other_loss": TrainingDynamics(_data={}),
            }
        )
        update_with_result = Result(sub_losses=update_with_registry)

        to_update_result.update(update_with_result)
        expected_recon_dyn = TrainingDynamics(_data=expected_result)
        expected_varloss_dyn = TrainingDynamics(_data=to_update)
        expected_moreloss_dyn = TrainingDynamics(_data={})
        expected_other_dyn = TrainingDynamics(_data={})
        expected_loss_registry = LossRegistry(
            _losses={
                "recon_loss": expected_recon_dyn,
                "var_loss": expected_varloss_dyn,
                "more_loss": expected_moreloss_dyn,
                "other_loss": expected_other_dyn,
            }
        )
        expected_result = Result(sub_losses=expected_loss_registry)
        assert expected_result.sub_losses == to_update_result.sub_losses
