import pytest
from unittest.mock import Mock
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._result import Result


class MockConreteTrainer(BaseTrainer):
    def train(self):
        pass


class TestBaseTrainerUnit:

    def test_trainsset_validation(self):
        with pytest.raises(TypeError):
            MockConreteTrainer(
                trainset=None,
                validset=Mock(spec=BaseDataset),
                result=Mock(spec=Result),
                config=Mock(spec=DefaultConfig),
                called_from="MockConcreteTrainer",
            )

    def test_trainset_type_validation(self):
        with pytest.raises(TypeError):
            MockConreteTrainer(
                trainset=Mock(spec=DefaultConfig),
                validset=None,
                result=Mock(spec=Result),
                config=Mock(spec=DefaultConfig),
                called_from="MockConcreteTrainer",
            )

    def test_validset_type_validation(self):
        with pytest.raises(TypeError):
            MockConreteTrainer(
                trainset=Mock(spec=BaseDataset),
                validset=Mock(spec=DefaultConfig),
                result=Mock(spec=Result),
                config=Mock(spec=DefaultConfig),
                called_from="MockConcreteTrainer",
            )

    def test_no_config(self):
        with pytest.raises(TypeError):
            MockConreteTrainer(
                trainset=Mock(spec=BaseDataset),
                validset=Mock(spec=BaseDataset),
                result=Mock(spec=Result),
                config=None,
                called_from="MockConcreteTrainer",
            )
