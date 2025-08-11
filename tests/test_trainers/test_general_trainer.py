import pytest
import numpy as np
import torch
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._losses import VanillixLoss
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.modeling._vanillix_architecture import VanillixArchitecture


class TestGeneralTrainerIntegration:
    @pytest.fixture
    def default_config(self):
        return DefaultConfig(epochs=1, checkpoint_interval=1, device="cpu")

    @pytest.fixture
    def train_dataset(self, default_config):
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        return NumericDataset(data, config=default_config)

    @pytest.fixture
    def valid_dataset(self, default_config):
        data = torch.tensor(
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
        )
        return NumericDataset(data, config=default_config)

    @pytest.fixture
    def empty_result(self):
        return Result()

    @pytest.fixture
    def filled_result(self, train_dataset, valid_dataset):
        result = Result()
        # fill with preprocessed_data and datasets (that might happen before training)
        result.preprocessed_data = torch.tensor([1, 2, 3])
        result.datasets = DatasetContainer(train=train_dataset, valid=valid_dataset)
        return result

    @pytest.fixture
    def general_trainer(
        self, train_dataset, valid_dataset, default_config, filled_result
    ):
        return GeneralTrainer(
            trainset=train_dataset,
            validset=valid_dataset,
            result=filled_result,
            config=default_config,
            model_type=VanillixArchitecture,
            loss_type=VanillixLoss,
        )

    def test_train(self, general_trainer):
        result = general_trainer.train()
        assert result is not None, "Training should return a Result object."
        assert len(result.losses.get(split="train")) == len(
            result.losses.get(split="valid")
        )

    def test_result_not_overwritten(self, general_trainer, filled_result):

        before_preprocessed_data = filled_result.preprocessed_data
        result = general_trainer.train()
        assert (
            result.preprocessed_data is not None
        ), "Preprocessed data should not overwrite."
        assert (
            result.preprocessed_data is before_preprocessed_data
        ), "Preprocessed data should not overwrite."

    @pytest.mark.parametrize("devices", ["cpu", "cuda"])
    def test_reproducible(self, devices, train_dataset, valid_dataset):
        # if device not available, skip tes
        if not torch.cuda.is_available() and devices == "cuda":
            pytest.skip("CUDA not available.")
        if not torch.backends.mps.is_available and devices == "mps":
            pytest.skip("MPS not available.")
        config = DefaultConfig(
            device=devices, epochs=3, checkpoint_interval=1, reproducible=True
        )
        general_trainer = GeneralTrainer(
            trainset=train_dataset,
            validset=valid_dataset,
            result=Result(),
            config=config,
            model_type=VanillixArchitecture,
            loss_type=VanillixLoss,
        )
        result1 = general_trainer.train()
        train_loss1 = result1.losses.get(split="train")
        reconstructed_data1 = result1.reconstructions.get(split="train")
        general_trainer = GeneralTrainer(
            trainset=train_dataset,
            validset=valid_dataset,
            result=Result(),
            config=config,
            model_type=VanillixArchitecture,
            loss_type=VanillixLoss,
        )
        result2 = general_trainer.train()
        train_losses2 = result2.losses.get(split="train")
        reconstructed_data2 = result2.reconstructions.get(split="train")
        assert np.array_equal(
            train_loss1, train_losses2
        ), "Training should be reproducible."
        assert np.array_equal(
            reconstructed_data1, reconstructed_data2
        ), "Reconstruction should be reproducible."
