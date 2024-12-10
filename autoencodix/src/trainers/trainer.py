from typing import Optional, Union
from autoencodix.src.utils._result import Result
from autoencodix.src.utils.default_config import DefaultConfig
from autoencodix.src.modeling._vanillix_architecture import VanillixArchitecture
from autoencodix.src.data._numeric_dataset import NumericDataset
from autoencodix.src.base._base_dataset import BaseDataset


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
        config: Optional[Union[DefaultConfig, None]],
    ) -> Result:
        """
        Train the model
        """
        if not isinstance(train, (NumericDataset, BaseDataset)):
            raise TypeError(
                f"Expected train type to be NumericDataset or BaseDataset, got {type(train)}."
            )
        self.input_dim = train.get_input_dim()
        self.model = VanillixArchitecture(config=config, input_dim=self.input_dim)
        self._result.model = self.model
        return result
