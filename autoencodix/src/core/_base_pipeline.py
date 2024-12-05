import numpy as np
import pandas as pd
import abc
from typing import Dict, Optional, Union
from anndata import AnnData #type: ignore
import torch
from autoencodix.src.preprocessing import Preprocessor
from ._default_config import DefaultConfig


class BasePipeline(abc.ABC):
    """
    Abstract base class defining the interface for all models.

    This class provides the methods for preprocessing, training, predicting,
    visualizing, and evaluating models. Subclasses should implement specific
    behavior for each of these steps.

    Attributes
    ----------
    config : DefaultConfig
        The configuration for the model.
    data : Union[np.ndarray, AnnData, pd.DataFrame]
    preprocessor : Optional[Preprocessor]
        The preprocessor used for data preprocessing.
    predictor : Optional[None]
        The predictor used for making predictions.
    visualizer : Optional[None]
        The visualizer used for visualizing results.
    evaluator : Optional[None]
        The evaluator used for evaluating model performance.
    preprocessed_data : Optional[np.ndarray]
        The data after preprocessing.
    predictions : Optional[np.ndarray]
        The model predictions.
    evaluation_results : Optional[Dict[str, float]]
        The evaluation metrics.
    model : torch.nn.Module
        The model instance.
    """

    def __init__(
        #TODO adjust attributes and methods according to the changes in vanilix class
        self,
        data: Union[pd.DataFrame, AnnData, np.ndarray],
        config: Optional[DefaultConfig] = None,
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
        self.config = config
        self.x: Optional[Union[np.ndarray, torch.Tensor]] = None

        self.preprocessor: Preprocessor = Preprocessor()
        self.predictor: Optional[None] = None
        self.visualizer: Optional[None] = None
        self.evaluator: Optional[None] = None

        self.predictions: Optional[np.ndarray] = None
        self.evaluation_results: Optional[Dict[str, float]] = None
        self.model: Optional[torch.nn.Module] = None

    def preprocess(self) -> None:
        """
        Preprocess input data.

        Parameters
        ----------
        data : np.ndarray
            The input data to preprocess.

        Returns
        -------
        np.ndarray
            The preprocessed data.

        Raises
        ------
        NotImplementedError
            If the preprocessor is not initialized.
        """
        if not self.preprocessor:
            raise NotImplementedError("Preprocessor not initialized")

        self.x = self.preprocessor.preprocess(self.data)

    def fit(self) -> None:
        """
        Train the model on preprocessed data.

        Raises
        ------
        NotImplementedError
            If the trainer is not initialized or if data has not been preprocessed.
        """
        if not self.trainer or self.preprocessed_data is None:
            raise NotImplementedError(
                "Trainer not initialized or data not preprocessed"
            )

        self.trainer.fit(self.preprocessed_data)

    def predict(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        data : Optional[np.ndarray], optional
            The input data for prediction (default is None, in which case preprocessed data is used).

        Returns
        -------
        np.ndarray
            The prediction results.

        Raises
        ------
        NotImplementedError
            If the predictor is not initialized.
        ValueError
            If no data is available for prediction.
        """
        if not self.predictor:
            raise NotImplementedError("Predictor not initialized")

        input_data = data if data is not None else self.preprocessed_data

        if input_data is None:
            raise ValueError("No data available for prediction")

        self.predictions = self.predictor.predict(input_data)
        return self.predictions

    def visualize(self) -> None:
        """
        Visualize model results.

        Raises
        ------
        NotImplementedError
            If the visualizer is not initialized.
        ValueError
            If no data is available for visualization.
        """
        if not self.visualizer:
            raise NotImplementedError("Visualizer not initialized")

        if self.preprocessed_data is None:
            raise ValueError("No data available for visualization")

        self.visualizer.visualize(self.preprocessed_data, self.predictions)

    def evaluate(self, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth data for evaluation.

        Returns
        -------
        Dict[str, float]
            The evaluation metrics.

        Raises
        ------
        NotImplementedError
            If the evaluator is not initialized.
        ValueError
            If no predictions are available for evaluation.
        """
        if not self.evaluator:
            raise NotImplementedError("Evaluator not initialized")

        if self.predictions is None:
            raise ValueError("No predictions available for evaluation")

        self.evaluation_results = self.evaluator.evaluate(
            ground_truth, self.predictions
        )
        return self.evaluation_results

    def run(self, data: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> None:
        """
        Run the entire model pipeline.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        ground_truth : Optional[np.ndarray], optional
            The ground truth for evaluation (default is None).
        """
        self.preprocess()
        self.fit()
        self.predict()
        self.visualize()

        if ground_truth is not None:
            self.evaluate(ground_truth)
