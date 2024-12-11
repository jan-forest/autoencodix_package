from typing import Optional, Union
from src.autoencodix.utils._result import Result
from src.autoencodix.utils.default_config import DefaultConfig
from src.autoencodix.modeling._vanillix_architecture import VanillixArchitecture
from src.autoencodix.data._numeric_dataset import NumericDataset
from src.autoencodix.base._base_dataset import BaseDataset


class Trainer:
    """
    Handles model training process
    """

    def __init__(self):
        self._result = Result()
        pass

    def train(
        self,
        train: Optional[Union[NumericDataset, BaseDataset]],
        valid: Optional[Union[NumericDataset, BaseDataset]],
        result: Result,
        config: Optional[Union[None, DefaultConfig]],
    ) -> Result:
        """
        Train the model
        """
        if not isinstance(train, (NumericDataset, BaseDataset)):
            raise TypeError(
                f"Expected train type to be NumericDataset or BaseDataset, got {type(train)}."
            )
        if config is None:
            raise ValueError("Config cannot be None.")
        epochs = config.epochs
        print(f"Training for {epochs} epochs.")
        self.input_dim = train.get_input_dim()
        self.model = VanillixArchitecture(config=config, input_dim=self.input_dim)

        self._result.model = self.model
        
        return result
