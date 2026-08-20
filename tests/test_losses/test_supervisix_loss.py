import pytest
import pandas as pd
import torch

from autoencodix.utils._result import Result
from autoencodix.utils._losses import SupervisixLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._datasetcontainer import DatasetContainer


class TestSupervisixClassLosses:
    """
    Unit tests for the class separation loss and the class cohesion loss in 
    isolation. 
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
            loss_reduction="sum"
        )

    @pytest.fixture
    def default_config_mean(self):
        return DefaultConfig(
            epochs=1,
            checkpoint_interval=1,
            device="cpu",
            class_param=self.CLASS_COL,
            latent_dim=self.LATENT_DIM,
            loss_reduction="mean"
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
    def train_dataset_mean(self, default_config_mean):
        # two samples of class "A", two of class "B"
        return self._make_dataset(default_config_mean, ["A", "A", "B", "B"])

    @pytest.fixture
    def single_class_train_dataset(self, default_config):
        # two samples of class "A", two of class "B"
        return self._make_dataset(default_config, ["A", "A", "A", "A"])

    @pytest.fixture
    def multi_class_train_dataset(self, default_config):
        return self._make_dataset(default_config, ["A", "A", "B", "B", "C", "C"])

    @pytest.fixture
    def valid_dataset(self, default_config):
        return self._make_dataset(default_config, ["A", "B"])

    @pytest.fixture
    def filled_result(self, train_dataset, valid_dataset):
        result = Result()
        result.datasets = DatasetContainer(train=train_dataset, valid=valid_dataset)
        return result

    @pytest.fixture
    def supervisix_loss(self, default_config):
        return SupervisixLoss(config=default_config)

    @pytest.fixture
    def supervisix_loss_mean(self, default_config_mean):
        return SupervisixLoss(config=default_config_mean)

    def _model_output(self, latentspace: torch.Tensor) -> ModelOutput:
        return ModelOutput(
            reconstruction=torch.zeros_like(latentspace),
            latentspace=latentspace,
        )

    # -- _distance_latent -------------------------------------------------
    
    def test_distance_latent(self, supervisix_loss, supervisix_loss_mean):
        """
        Test that distance_latent computes the correct distance 
        between a multidimensional tensor and a one dimensional tensor,
        including correct reduction.
        """
        tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor_b = torch.tensor([0.0, 1.0])

        distance_sum = supervisix_loss._distance_latent(latent_a=tensor_a, latent_b=tensor_b)
        distance_mean = supervisix_loss_mean._distance_latent(latent_a=tensor_a, latent_b=tensor_b)

        assert torch.equal(distance_sum, torch.tensor(4.0))
        assert torch.equal(distance_mean, torch.tensor(2.0))

    # -- _compute_class_separation_loss -------------------------------------------------

    # TODO: Add validation dataset to tests

    def test_compute_class_separation_loss_two_classes(self, 
                                                       train_dataset,
                                                       train_dataset_mean,
                                                       supervisix_loss,
                                                       supervisix_loss_mean):
        """
        Test that the class separation loss correctly computes the loss for a batch with 
        two classes.
        """
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1,2,3]
        last_epoch_class_means = {"A": torch.tensor([2.0, 2.0]), "B": torch.tensor([15.0, 15.0])}

        loss_mean = supervisix_loss_mean._compute_class_separation_loss(model_output=model_output,
                                                                   sample_ids=sample_ids,
                                                                   metadata=train_dataset_mean.metadata,
                                                                   last_epoch_class_means=last_epoch_class_means)
        loss_sum = supervisix_loss._compute_class_separation_loss(model_output=model_output,
                                                                       sample_ids=sample_ids,
                                                                       metadata=train_dataset.metadata,
                                                                       last_epoch_class_means=last_epoch_class_means)

        assert torch.equal(loss_mean, torch.tensor(-13.0))
        assert torch.equal(loss_sum, torch.tensor(-26.0))

    def test_compute_class_separation_loss_first_epoch(self,
                                                        train_dataset,
                                                        supervisix_loss):
        """
        Test that _compute_class_separations_loss returns 0.0 on first epoch.
        """
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1,2,3]
        last_epoch_class_means = {}

        loss = supervisix_loss._compute_class_separation_loss(model_output=model_output,
                                                              sample_ids=sample_ids,
                                                              metadata=train_dataset.metadata,
                                                              last_epoch_class_means=last_epoch_class_means)

        assert torch.equal(loss, torch.tensor(0.0))

    def test_compute_class_separation_loss_single_class_batch(self,
                                                              train_dataset,
                                                              train_dataset_mean,
                                                              supervisix_loss,
                                                              supervisix_loss_mean):
        """
        Test that class separation loss is handled correctly when a batch contains only one class.
        For either case a) last epoch mean for class B is saved or b) there is no last epoch class mean
        for class B saved.
        """
        # rows 0,1 -> class A (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1]
        last_epoch_class_means = {"A": torch.tensor([2.0, 2.0]), "B": torch.tensor([15.0, 15.0])}

        loss_sum = supervisix_loss._compute_class_separation_loss(model_output=model_output,
                                                                  sample_ids=sample_ids,
                                                                  metadata=train_dataset.metadata,
                                                                  last_epoch_class_means=last_epoch_class_means)
        
        loss_mean = supervisix_loss_mean._compute_class_separation_loss(model_output=model_output,
                                                                        sample_ids=sample_ids,
                                                                        metadata=train_dataset_mean.metadata,
                                                                        last_epoch_class_means=last_epoch_class_means)

        assert torch.equal(loss_sum, torch.tensor(-13.0))
        assert torch.equal(loss_mean, torch.tensor(-6.5))    

    def test_compute_class_separation_loss_three_classes(self,
                                                         multi_class_train_dataset,
                                                         supervisix_loss,
                                                         supervisix_loss_mean):
        """
        Test if class separation loss is computed correctly when there are more than two 
        classes in the dataset.
        """
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0], [5.0, 5.0], [7.0, 7.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1,2,3,4,5]
        last_epoch_class_means = {"A": torch.tensor([2.0, 2.0]), "B": torch.tensor([15.0, 15.0]), "C": torch.tensor([6.0, 6.0])}

        loss_sum = supervisix_loss._compute_class_separation_loss(model_output=model_output,
                                                                  sample_ids=sample_ids,
                                                                  metadata=multi_class_train_dataset.metadata,
                                                                  last_epoch_class_means=last_epoch_class_means)
        
        loss_mean = supervisix_loss_mean._compute_class_separation_loss(model_output=model_output,
                                                                        sample_ids=sample_ids,
                                                                        metadata=multi_class_train_dataset.metadata,
                                                                        last_epoch_class_means=last_epoch_class_means)

        assert torch.isclose(loss_sum, torch.tensor(-52.0/3))
        assert torch.isclose(loss_mean, torch.tensor(-26.0/3))        

    # -- _compute_class_cohesion_loss -------------------------------------------------

    def test_compute_class_cohesion_loss_two_classes(self, 
                                                     train_dataset,
                                                     train_dataset_mean,
                                                     supervisix_loss,
                                                     supervisix_loss_mean):
        """
        Test that the class cohesion loss is computed correctly for two classes in a batch.
        """
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1,2,3]
        last_epoch_class_means = {"A": torch.tensor([2.0, 2.0]), "B": torch.tensor([15.0, 15.0])}

        loss_mean = supervisix_loss_mean._compute_class_cohesion_loss(model_output=model_output,
                                                                 sample_ids=sample_ids,
                                                                 metadata=train_dataset_mean.metadata,
                                                                 last_epoch_class_means=last_epoch_class_means)
        loss_sum = supervisix_loss._compute_class_cohesion_loss(model_output=model_output,
                                                                     sample_ids=sample_ids,
                                                                     metadata=train_dataset.metadata,
                                                                     last_epoch_class_means=last_epoch_class_means)

        assert torch.equal(loss_mean, torch.tensor(3.0))
        assert torch.equal(loss_sum, torch.tensor(6.0))

    def test_compute_class_cohesion_loss_first_epoch(self,
                                                     train_dataset,
                                                     supervisix_loss):
        """
        Test that _compute_class_cohesion_loss returns 0.0 on first epoch.
        """
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1,2,3]
        last_epoch_class_means = {}

        loss = supervisix_loss._compute_class_cohesion_loss(model_output=model_output,
                                                            sample_ids=sample_ids,
                                                            metadata=train_dataset.metadata,
                                                            last_epoch_class_means=last_epoch_class_means)

        assert torch.equal(loss, torch.tensor(0.0))

    def test_compute_class_cohesion_loss_single_class_batch(self,
                                                              train_dataset,
                                                              train_dataset_mean,
                                                              supervisix_loss,
                                                              supervisix_loss_mean):
        """
        Test that class cohesion loss is handled correctly when a batch contains only one class.
        For either case a) last epoch mean for class B is saved or b) there is no last epoch class mean
        for class B saved.
        """
        # rows 0,1 -> class A (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1]
        last_epoch_class_means = {"A": torch.tensor([2.0, 2.0]), "B": torch.tensor([15.0, 15.0])}

        loss_sum = supervisix_loss._compute_class_cohesion_loss(model_output=model_output,
                                                                  sample_ids=sample_ids,
                                                                  metadata=train_dataset.metadata,
                                                                  last_epoch_class_means=last_epoch_class_means)
        
        loss_mean = supervisix_loss_mean._compute_class_cohesion_loss(model_output=model_output,
                                                                        sample_ids=sample_ids,
                                                                        metadata=train_dataset_mean.metadata,
                                                                        last_epoch_class_means=last_epoch_class_means)

        assert torch.equal(loss_sum, torch.tensor(1.0))
        assert torch.equal(loss_mean, torch.tensor(0.5))    

    def test_compute_class_cohesion_loss_three_classes(self,
                                                         multi_class_train_dataset,
                                                         supervisix_loss,
                                                         supervisix_loss_mean):
        """
        Test if class cohesion loss is computed correctly when there are more than two 
        classes in the dataset.
        """
        # rows 0,1 -> class A ; rows 2,3 -> class B (matches train_dataset fixture order)
        latentspace = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0], [5.0, 5.0], [7.0, 7.0]]
        )
        model_output = self._model_output(latentspace)
        sample_ids = [0,1,2,3,4,5]
        last_epoch_class_means = {"A": torch.tensor([2.0, 2.0]), "B": torch.tensor([15.0, 15.0]), "C": torch.tensor([6.0, 6.0])}

        loss_sum = supervisix_loss._compute_class_cohesion_loss(model_output=model_output,
                                                                  sample_ids=sample_ids,
                                                                  metadata=multi_class_train_dataset.metadata,
                                                                  last_epoch_class_means=last_epoch_class_means)
        
        loss_mean = supervisix_loss_mean._compute_class_cohesion_loss(model_output=model_output,
                                                                        sample_ids=sample_ids,
                                                                        metadata=multi_class_train_dataset.metadata,
                                                                        last_epoch_class_means=last_epoch_class_means)

        assert torch.isclose(loss_sum, torch.tensor((14.0 / 3)))
        assert torch.isclose(loss_mean, torch.tensor((7.0 / 3)))