import abc
from typing import Optional, Union, Type

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore

from autoencodix.src.base._base_dataset import BaseDataset
from autoencodix.src.data._datasetcontainer import DataSetContainer
from autoencodix.src.data._datasplitter import DataSplitter
from autoencodix.src.evaluate.evaluate import Evaluator
from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.trainers.predictor import Predictor
from autoencodix.src.trainers.trainer import Trainer
from autoencodix.src.utils._result import Result
from autoencodix.src.utils.default_config import DefaultConfig, config_method
from autoencodix.src.visualize.visualize import Visualizer


# TODO test`
class BasePipeline(abc.ABC):
    """
    Abstract base class defining the interface for all models.

    This class provides the methods for preprocessing, training, predicting,
    visualizing, and evaluating models. Subclasses should implement specific
    behavior for each of these steps.

    Attributes
    ----------
    config : DefaultConfig
        The configuration for the pipeline. Handles preprocessing, training, and
        other settings.
    data : Union[np.ndarray, AnnData, pd.DataFrame]
    _features : Optional[np.ndarray]
        The data after preprocessing.
    _preprocessor : Optional[Preprocessor]
        The preprocessor used for data preprocessing.
    _datasets: Optional[DataSetContainer]
    _data_splitter : Optional[DataSplitter]:
        returns train, valid, test indices
    _trainer : Optional[Trainer]
    _predictor : Optional[Predictor]
        The predictor used for making predictions.
    _evaluator : Optional[Evaluator]
        The evaluator used for evaluating model performance.
    _visualizer : Optional[Visualizer]
        The visualizer used for visualizing results.
        result: Result

    Methods
    -------
    _build_dataset(*kwargs):
        Calls a DataSplitter instance to obtain train, valid and test indicies
        Updates the self._datset attribute with the train, valid and test datasets
        Datsets are a subclass of torch.utils.data.Dataset and can be different subclasses
        depending on the Pipeline.
    preprocess(*kwargs):
        Calls the Preprocessor instance to preprocess the data. UPdates the self._preprocessed_data
        attribute.
    fit(*kwargs):
        Calls the Trainer instance to train the model on the preprocessed data.
        Updates the self._model attribute and result attribute.
    predict(*kwargs):
        Calls the Predictor instance to run inference with the test data on the trained model.
        If user inputs data, it preprocesses the data and runs inference.
        Updates the result attribute.
    evaluate(*kwargs):
        Calls the Evaluator instance to evaluate the model performane in downstream tasks.
        Updates the result attribute.
    visualize(*kwargs):
        Calls the Visualizer instance to visualize all relevant data in the result attribute.
    show_result():
        Helper Function to directly show the visualized results.
    run():
        Runs the entire pipeline in the following order: preprocess, fit, predict, evaluate, visualize.
        Updates the result attribute.
    """
    # needed for the predict function to be able to use a child of BaseDataset in subclasses
    dataset_class: Type[BaseDataset] = BaseDataset

    def __init__(
        self,
        data: Union[pd.DataFrame, AnnData, np.ndarray],
        config: Optional[Union[None, DefaultConfig]] = None,
        dataset_splitter: Optional[DataSplitter] = None,
        **kwargs,
    ):
        """
        Initialize the model interface.

        Parameters
        ----------
        config : DefaultConfig, optional
            The configuration dictionary for the model.
        """
        if not isinstance(data, (np.ndarray, AnnData, pd.DataFrame)):
            raise TypeError(
                f"Expected data type to be one of np.ndarray, AnnData, or pd.DataFrame, got {type(data)}."
            )
        self.data: Union[np.ndarray, AnnData, pd.DataFrame] = data
        self.config = DefaultConfig()
        if config:
            if not isinstance(config, DefaultConfig):
                raise TypeError(
                    f"Expected config type to be DefaultConfig, got {type(config)}."
                )
            self.config = config
        self._data_splitter = DataSplitter()
        if dataset_splitter:
            if not isinstance(dataset_splitter, DataSplitter):
                raise TypeError(
                    f"Expected dataset_splitter type to be DataSplitter, got {type(dataset_splitter)}."
                )
            self._data_splitter = dataset_splitter

        self._preprocessor: Optional[Preprocessor]
        self._features: Optional[torch.Tensor]
        self._datasets: Optional[DataSetContainer]
        self._trainer: Optional[Trainer]
        self._predictor: Optional[Predictor]
        self._visualizer: Optional[Visualizer]
        self._evaluator: Optional[Evaluator]
        self.result: Result

    def _build_datasets(self) -> None:
        """
        Build datasets for training, validation, and testing.

        Raises
        ------
        NotImplementedError
            If the data splitter is not initialized.
        ValueError
            If self._features is None.
        """
        if self._data_splitter is None:
            raise NotImplementedError("Data splitter not initialized")

        if self._features is None:
            raise ValueError("No data available for splitting")

        split_indices = self._data_splitter.split(self._features)
        train_ids, valid_ids, test_ids = (
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
        )

        self._datasets = DataSetContainer(
            train=BaseDataset(data=self._features[train_ids]),
            valid=BaseDataset(data=self._features[valid_ids]),
            test=BaseDataset(data=self._features[test_ids]),
        )

    # config parameter will be self.config if not provided, decorator will handle this
    @config_method
    def preprocess(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Takes the user input data and filters, norrmalizes and cleans the data.
        Populates the self._features attribute with the preprocessed data as a numpy array.
        Calls the _build_datasets method to build the datasets for training, validation and testing.

        """
        if self._preprocessor is None:
            raise NotImplementedError("Preprocessor not initialized")

        self._features = self._preprocessor.preprocess(self.data)
        self._build_datasets()
        self.result.preprocessed_data = self._features
        self.result.datasets = self._datasets

    @config_method
    def fit(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Train the model on preprocessed data.
        """
        if self._features is None:
            raise ValueError("No data available for training, please preprocess first")

        if self._trainer is None:
            raise NotImplementedError("_trainer not initialized")
        if self._datasets is None:
            raise ValueError(
                "Datasets not built. Please run the preprocess method first."
            )

        trainer_result = self._trainer.train(
            train=self._datasets.train,
            valid=self._datasets.valid,
            result=self.result,
            config=config,
        )
        self.result.update(trainer_result)

    @config_method
    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame, AnnData] = None,
        config: Optional[Union[None, DefaultConfig]] = None,
        **kwargs,
    ) -> None:
        """
        Run inference with the test data on the trained model.
        If user inputs data, it preprocesses the data and runs inference.
        """
        if self._preprocessor is None:
            raise NotImplementedError("Preprocessor not initialized")
        if self._predictor is None:
            raise NotImplementedError("Predictor not initialized")
        if self._datasets is None:
            raise NotImplementedError(
                "Datasets not built. Please run the fit method first."
            )
        if self.result.model is None:
            raise NotImplementedError(
                "Model not trained. Please run the fit method first"
            )

        if data:
            processed_data = self._preprocessor.preprocess(data)
            input_data = self.dataset_class(data=processed_data)
            predictor_results = self._predictor.predict(data=input_data)
        else:
            predictor_results = self._predictor.predict(data=self._datasets.test)
        self.result.update(predictor_results)

    @config_method
    def evaluate(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        pass

    @config_method
    def visualize(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Visualize model results.

        Raises
        ------
        NotImplementedError
            If the visualizer is not initialized.
        ValueError
            If no data is available for visualization.
        """
        if self._visualizer is None:
            raise NotImplementedError("Visualizer not initialized")

        if self._features is None:
            raise ValueError("No data available for visualization")

        self._visualizer.visualize(self.result)

    def show_result(self) -> None:
        pass

    def run(self) -> Result:
        """
        Run the entire model pipeline (preprocess, fit, predict, evaluate, visualize).
        Populates the result attribute and returns it.
        """
        self.preprocess()
        self.fit()
        self.predict()
        self.evaluate()
        self.visualize()
        return self.result
