# Test written by Claude, reviewed and edited by Kristina Müller
import pytest
import pandas as pd
import torch

from autoencodix.trainers._supervisix_trainer import SupervisixTrainer
from autoencodix.utils._result import Result
from autoencodix.utils._losses import SupervisixLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.modeling._varix_architecture import VarixArchitecture


class TestSupervisixTrainerHooks:
    """
    Unit tests for the three supervisix-specific hooks in isolation.

    Each hook is called directly with hand-built inputs (rather than by
    driving the full `.train()` loop), so the expected per-class means can
    be asserted exactly and the tests don't depend on the model/optimizer
    doing anything in particular.
    """

    CLASS_COL = "celltype"
    LATENT_DIM = 2

    @pytest.fixture
    def default_config(self):
        return DefaultConfig(
            epochs=1,
            checkpoint_interval=1,
            device="cpu",
            class_param=self.CLASS_COL,
            latent_dim=self.LATENT_DIM,
        )

    def _make_dataset(self, config, class_labels):
        n = len(class_labels)
        data = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
        metadata = pd.DataFrame({self.CLASS_COL: class_labels})
        return NumericDataset(data, config=config, metadata=metadata)

    @pytest.fixture
    def train_dataset(self, default_config):
        # two samples of class "A", two of class "B"
        return self._make_dataset(default_config, ["A", "A", "B", "B"])

    @pytest.fixture
    def single_class_train_dataset(self, default_config):
        # two samples of class "A", two of class "B"
        return self._make_dataset(default_config, ["A", "A", "A", "A"])

    @pytest.fixture
    def valid_dataset(self, default_config):
        return self._make_dataset(default_config, ["A", "B"])

    @pytest.fixture
    def filled_result(self, train_dataset, valid_dataset):
        result = Result()
        result.datasets = DatasetContainer(train=train_dataset, valid=valid_dataset)
        return result

    @pytest.fixture
    def trainer(self, train_dataset, valid_dataset, default_config, filled_result):
        return SupervisixTrainer(
            trainset=train_dataset,
            validset=valid_dataset,
            result=filled_result,
            config=default_config,
            model_type=VarixArchitecture,
            loss_type=SupervisixLoss,
            ontologies=None,
        )

    @pytest.fixture
    def single_class_trainer(self, single_class_train_dataset, valid_dataset, default_config, filled_result):
        return SupervisixTrainer(
            trainset=single_class_train_dataset,
            validset=valid_dataset,
            result=filled_result,
            config=default_config,
            model_type=VarixArchitecture,
            loss_type=SupervisixLoss,
            ontologies=None,
        )

    def _model_output(self, latentspace: torch.Tensor) -> ModelOutput:
        return ModelOutput(
            reconstruction=torch.zeros_like(latentspace),
            latentspace=latentspace,
        )

    # -- supervisix_capture_hook -------------------------------------------------

    def test_capture_hook_computes_per_class_batch_mean(self, trainer):
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]]
        )
        model_output = self._model_output(latentspace)
        batch_class_means: dict = {}

        trainer.supervisix_capture_hook(
            model_output=model_output,
            sample_ids=(0, 1, 2, 3),
            dataset_type="train",
            batch_class_means=batch_class_means,
        )

        assert torch.equal(batch_class_means["A"], torch.tensor([[2.0, 2.0]]))
        assert torch.equal(batch_class_means["B"], torch.tensor([[15.0, 15.0]]))

    def test_capture_hook_stacks_across_batches(self, trainer):
        batch_class_means: dict = {}

        first_latents = torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]])
        trainer.supervisix_capture_hook(
            model_output=self._model_output(first_latents),
            sample_ids=(0, 1, 2, 3),
            dataset_type="train",
            batch_class_means=batch_class_means,
        )

        second_latents = torch.tensor([[5.0, 5.0], [7.0, 7.0], [30.0, 30.0], [40.0, 40.0]])
        trainer.supervisix_capture_hook(
            model_output=self._model_output(second_latents),
            sample_ids=(0, 1, 2, 3),
            dataset_type="train",
            batch_class_means=batch_class_means
        )

        # second call should stack the new per-batch mean onto the first, not overwrite it
        assert batch_class_means["A"].shape == (2, self.LATENT_DIM)
        assert torch.equal(batch_class_means["A"][0], torch.tensor([2.0, 2.0]))
        assert torch.equal(batch_class_means["A"][1], torch.tensor([6.0, 6.0]))

    def test_capture_hook_only_adds_new_class_if_not_present(self, single_class_trainer, trainer):
        batch_class_means: dict = {}

        first_latents = torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]])
        single_class_trainer.supervisix_capture_hook(
            model_output=self._model_output(first_latents),
            sample_ids=(0, 1, 2, 3),
            dataset_type="train",
            batch_class_means=batch_class_means,
        )

        # First call should add only class a to the class_batch_means dict
        assert "A" in batch_class_means
        assert "B" not in batch_class_means
        assert batch_class_means["A"].shape == (1, self.LATENT_DIM)
        assert torch.equal(batch_class_means["A"][0], torch.tensor([2.0, 2.0]))

        second_latents = torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]])
        trainer.supervisix_capture_hook(
            model_output=self._model_output(second_latents),
            sample_ids=(0, 1, 2, 3),
            dataset_type="train",
            batch_class_means=batch_class_means,
        )

        # Second call should add class B to the batch_class_means dict and update class A
        assert "B" in batch_class_means
        assert batch_class_means["A"].shape == (2, self.LATENT_DIM)
        assert torch.equal(batch_class_means["A"][0], torch.tensor([2.0, 2.0]))
        assert torch.equal(batch_class_means["A"][1], torch.tensor([2.0, 2.0]))
        assert batch_class_means["B"].shape == (1, self.LATENT_DIM)
        assert torch.equal(batch_class_means["B"][0], torch.tensor([15.0, 15.0]))

    # -- supervisix_update_hook -------------------------------------------------

    def test_update_hook_averages_batch_means_into_epoch_means(self, trainer):
        batch_class_means = {
            "A": torch.tensor([[2.0, 2.0], [6.0, 6.0]]),
            "B": torch.tensor([[15.0, 15.0], [35.0, 35.0]]),
        }

        trainer.supervisix_update_hook(batch_class_means, "train")

        assert torch.equal(trainer.epoch_class_means_train["A"], torch.tensor([4.0, 4.0]))
        assert torch.equal(trainer.epoch_class_means_train["B"], torch.tensor([25.0, 25.0]))
        # valid means must be untouched
        assert trainer.epoch_class_means_valid == {}

    def test_update_hook_writes_valid_means_for_valid_dataset_type(self, trainer):
        batch_class_means = {"A": torch.tensor([[1.0, 1.0], [3.0, 3.0]])}

        trainer.supervisix_update_hook(batch_class_means, "valid")

        assert torch.equal(trainer.epoch_class_means_valid["A"], torch.tensor([2.0, 2.0]))
        assert trainer.epoch_class_means_train == {}

    # -- supervisix_get_hook -------------------------------------------------

    def test_get_hook_returns_epoch_means_dict(self, trainer):
        # Pre-populate the epoch_class_means_train and epoch_class_means_valid
        trainer.epoch_class_means_train = {
            "A": torch.tensor([4.0, 4.0]),
            "B": torch.tensor([25.0, 25.0]),
        }
        trainer.epoch_class_means_valid = {
            "A": torch.tensor([2.0, 2.0]),
        }

        train_means = trainer.supervisix_get_hook("train")
        valid_means = trainer.supervisix_get_hook("valid")

        assert train_means == trainer.epoch_class_means_train
        assert valid_means == trainer.epoch_class_means_valid