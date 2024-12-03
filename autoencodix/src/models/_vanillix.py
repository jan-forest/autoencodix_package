from typing import Union, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from anndata import AnnData

from autoencodix.src.core._base_pipeline import BasePipeline
from autoencodix.src.core._default_config import DefaultConfig
from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.core._vanillix_architecture import VanillixArchitecture

# TODO decide wheter to have one ax class that gets a model object passed or ot have a pipeline
# class for each model, that performs the training, prediction, evaluation, etc.
#TODO: implement the following classes
class Trainer:
    pass


class DataSetBuilder:
    def build(self, x: np.ndarray) -> None:
        pass
@dataclass
class Results:
    # for train, validation, and test
    latentspaces: np.ndarray = field(default_factory=dict)
    reconstructions: np.ndarray = field(default_factory=dict)
    mus: np.ndarray = field(default_factory=dict)
    sigmas: np.ndarray = field(default_factory=dict)
    losses: np.ndarray = field(default_factory=dict)
    

    
class Vanillix(BasePipeline):
    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        config: Optional[DefaultConfig] = None,
    ):

        super().__init__(data, config)
        self.data = data
        self.x = None
        self.model = VanillixArchitecture(config=self.config)
        self.preprocessor = Preprocessor()
        self.trainer = Trainer()
        self.config = config
        self._dataset = None

    def _buil_dataset(self) -> None:
        """
        Uses the self.x np.ndarray to build a dataset object that can be passed
        to a PyTorch DataLoader. Populates the private attribute self._dataset.
        
        """
        # TODO decided how to handle, train, validation, test split
        # should user provide the split or should we handle it internally
        self._dataset = DataSetBuilder().build(self.x)
        pass

    def preprocess(self) -> None:
        """
        Takes the user input data and filters, norrmalizes and cleans the data.
        Populates the self.x attribute with the preprocessed data as a numpy array.

        """
        self.x = self.preprocessor.preprocess(self.data)

    def fit(self) -> None:
        """

        
        """
        # needs train and valid loader build from dataset
        # TODO decide wheter to handle this in the pipeline or in the trainer
        self.model = self.trainer.train(self.model, self._dataset)

    def predict(self):
        # TODO decide wheter to return prediciton, or save in results attribute
        pass
