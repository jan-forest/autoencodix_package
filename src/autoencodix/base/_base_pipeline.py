import abc
from typing import Optional, Union, Dict, Type, List

import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore
from ._base_dataset import BaseDataset
from ._base_trainer import BaseTrainer
from ._base_predictor import BasePredictor
from ._base_visualizer import BaseVisualizer
from ._base_preprocessor import BasePreprocessor
from autoencodix.data._datasetcontainer import DataSetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._utils import config_method


# TODO test`
# internal check done
class BasePipeline(abc.ABC):
    """
    Abstract base class defining the interface for all models.

    This class provides the methods for preprocessing, training, predicting,
    visualizing, and evaluating (former ml_task step) models.
    Subclasses should perform steps like the parent class
    but has custom attribute e.g Processor, Trainer, Predictor, Evaluator, Visualizer.

    Attributes
    ----------
    config : DefaultConfig
        The configuration for the pipeline. Handles preprocessing, training, and
        other settings.
    data : Union[np.ndarray, AnnData, pd.DataFrame]
        User input data.
    data_splitter : Optional[DataSplitter]:
        returns train, valid, test indices, user can provide custom splitter
    result : Result
        dataclass to store all results from the pipeline.
    _features : Optional[torch.Tensor]
        The preprocessed data, output of former make_data.
    _preprocessor : Optional[Preprocessor]
        The preprocessor ffilters, scales, matches and cleans the data.
        Specific implementations for this will be used in subclasses.
    _datasets: Optional[DataSetContainer]
        data in form of Dataset classes after splitting, basically self._features wrapped in a
        Dataset class and split into train, valid and test datasets.
    _trainer : Optional[Trainer]
        Trainer object with train method to train actual model weights.
        Each subclass will have its own Trainer class.
    _predictor : Optional[Predictor]
        TODO write docs when actual implementation is done (now only mock).
    _evaluator : Optional[Evaluator]
        TODO write docs when actual implementation is done (now only mock).
    _visualizer : Optional[Visualizer]
        TODO write docs when actual implementation is done (now only mock).
    _id: str:
        Identifier for the pipeline (here base). Used to identify the pipeline in the
        specific pipeline subclasses, when calling specfic implementations of the Trainer, etc.
    _dataset_class: Type[BaseDataset]
        Used to use the methods from the parent class in the child classes with the correct type.
    _trainer_class: Type[BaseTrainer]
        Used to use the methods from the parent class in the child classes with the correct type.


    Methods
    -------
    _build_dataset(*kwargs):
        Calls a DataSplitter instance to obtain train, valid and test indicies
        Updates the self._datset attribute with the train, valid and test datasets
        Datsets are a subclass of torch.utils.data.Dataset and can be different subclasses
        depending on the Pipeline.
    preprocess(*kwargs):
        Calls the Preprocessor instance to preprocess the data. Updates the self._features attribute.
        Populates the self.results attribute with self._features and self._datasets.
    fit(*kwargs):
        Calls the Trainer instance to train the model on the with training and validation data of self._datasets.
        Populates the self.results attribute with the trained model and training dynamics and results.
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

    def __init__(
        self,
        data: Union[pd.DataFrame, AnnData, np.ndarray, List[np.ndarray]],
        dataset_type: Type[BaseDataset],
        trainer_type: Type[BaseTrainer],
        datasplitter_type: Type[DataSplitter],
        preprocessor: BasePreprocessor,
        predictor: BasePredictor,
        visualizer: BaseVisualizer,
        result: Result,
        config: Optional[Union[None, DefaultConfig]] = DefaultConfig(),
        custom_split: Optional[Dict[str, np.ndarray]] = None,
        **kwargs,
    ) -> None:
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

        self._id = "base"
        self.data: Union[np.ndarray, AnnData, pd.DataFrame] = data
        self.config = config
        self._trainer_type = trainer_type
        self._preprocessor = preprocessor
        self._predictor = predictor
        self._visualizer = visualizer
        self._dataset_type = dataset_type
        self.result = result
        self._data_splitter = datasplitter_type(
            config=self.config, custom_splits=custom_split
        )

        if config:
            if not isinstance(config, DefaultConfig):
                raise TypeError(
                    f"Expected config type to be DefaultConfig, got {type(config)}."
                )
            self.config = config

        self._features: Optional[torch.Tensor]
        self._datasets: Optional[DataSetContainer]

        # config parameter will be self.config if not provided, decorator will handle this

    @config_method(valid_params={"config"})
    def preprocess(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Takes the user input data and filters, norrmalizes and cleans the data.
        Populates the self._features attribute with the preprocessed data as a numpy array.
        Calls the _build_datasets method to build the datasets for training, validation and testing.

        Parameters:
            config: DefaultConfig, optional (default: None)
                allows to pass a custom configuration for the preprocessing step.
            **kwargs:
                User can pass configuration parameters as single keyword arguments.
        Raises:
            NotImplementedError:
                If the preprocessor is not initialized.

        """
        if self._preprocessor is None:
            raise NotImplementedError("Preprocessor not initialized")

        self._datasets, self._features = self._preprocessor.preprocess(
            data=self.data,
            data_splitter=self._data_splitter,
            config=self.config,
            dataset_type=self._dataset_type,
        )
        self.result.preprocessed_data = self._features
        self.result.datasets = self._datasets

    @config_method(
        valid_params={
            "config",
            "batch_size",
            "epochs",
            "learning_rate",
            "n_workers",
            "device",
            "n_gpus",
            "gpu_strategy",
            "weight_decay",
            "reproducible",
            "global_seed",
            "reconstruction_loss",
            "checkpoint_interval",
        }
    )
    def fit(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Trains the model, based on the in the child class defined Trainer class.
        Populates the self.result attribute with the trained model and training dynamics and results.

        Parameters:
            config: DefaultConfig, optional (default: None)
                allows to pass a custom configuration for the training step.
            **kwargs:
                User can pass configuration parameters as single keyword arguments.
        Raises:
            ValueError:
                If no data is available for training.
            ValueError:
                If the datasets are not built. Please run the preprocess method first.

        """
        if self._features is None:
            raise ValueError("No data available for training, please preprocess first")

        if self._datasets is None:
            raise ValueError(
                "Datasets not built. Please run the preprocess method first."
            )

        self._trainer = self._trainer_type(
            trainset=self._datasets.train,
            validset=self._datasets.valid,
            result=self.result,
            config=config,
            called_from=self._id,
        )
        trainer_result = self._trainer.train()
        self.result.update(trainer_result)

    @config_method(valid_params={"config"})
    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame, AnnData] = None,
        config: Optional[Union[None, DefaultConfig]] = None,
        **kwargs,
    ) -> None:
        """
        Run inference with the test data on the trained model.
        If user inputs data, it preprocesses the data and runs inference.
        Populates the self.result attribute with the inference results.

        Parameters:
            data: Union[np.ndarray, pd.DataFrame, AnnData], optional (default: None)
                User input data to run inference on.
            config: DefaultConfig, optional (default: None)
                allows to pass a custom configuration for the prediction step.
            **kwargs:
                User can pass configuration parameters as single keyword arguments.
        Raises:
            NotImplementedError:
                If the preprocessor is not initialized.
            NotImplementedError:
                If the predictor is not initialized.
            NotImplementedError:
                If the datasets are not built. Please run the preprocess method first.
            NotImplementedError:
                If the model is not trained. Please run the fit method first.
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

        if data is not None:
            _, processed_data = self._preprocessor.preprocess(
                data=data,
                data_splitter=None,
                config=self.config,
                dataset_type=self._dataset_type,
                split=False,
            )
            input_data = self._dataset_type(data=processed_data, config=self.config)
            predictor_results = self._predictor.predict(data=input_data)
        else:
            predictor_results = self._predictor.predict(data=self._datasets.test)
        self.result.update(predictor_results)

    @config_method(valid_params={"config"})
    def evaluate(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        pass

    @config_method(valid_params={"config"})
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

    def run(
        self, data: Optional[Union[pd.DataFrame, np.ndarray, AnnData]] = None
    ) -> Result:
        """
        Run the entire model pipeline (preprocess, fit, predict, evaluate, visualize).
        When predict step should be run on user input data, the data parameter should be provided.
        (Overrides test data from the pipeline object)
        Populates the result attribute and returns it.

        Parameters:
            data: Union[pd.DataFrame, np.ndarray, AnnData], optional (default: None)

        Returns:
            Result:
                The result object containing all relevant data from the pipeline.

        """
        self.preprocess()
        self.fit()
        self.predict(data=data)
        self.evaluate()
        self.visualize()
        return self.result
