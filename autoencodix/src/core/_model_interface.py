import numpy as np
import abc
from typing import Dict, Any, Optional, Union
import torch
from autoencodix.preprocessing import Preprocessor


class ModelInterface(abc.ABC):
    """
    Abstract base class defining the interface for all models
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model interface

        Args:
            config (Union[ModelConfig, Dict[str, Any]], optional): Model configuration
        """
        self.config = config

        self.preprocessor: Optional[Preprocessor] = None
        self.predictor: Optional[None] = None
        self.visualizer: Optional[None] = None
        self.evaluator: Optional[None] = None

        self.preprocessed_data: Optional[np.ndarray] = None
        self.predictions: Optional[np.ndarray] = None
        self.evaluation_results: Optional[Dict[str, float]] = None
        self.model: torch.nn.Module = None

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data

        Args:
            data (np.ndarray): Input data to preprocess

        Returns:
            np.ndarray: Preprocessed data
        """
        if not self.preprocessor:
            raise NotImplementedError("Preprocessor not initialized")

        self.preprocessed_data = self.preprocessor.preprocess(data)
        return self.preprocessed_data

    def fit(self):
        """
        Train the model on preprocessed data
        """
        if not self.trainer or self.preprocessed_data is None:
            raise NotImplementedError(
                "Trainer not initialized or data not preprocessed"
            )

        self.trainer.fit(self.preprocessed_data)

    def predict(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions

        Args:
            data (Optional[np.ndarray], optional): Input data for prediction

        Returns:
            np.ndarray: Prediction results
        """
        if not self.predictor:
            raise NotImplementedError("Predictor not initialized")

        input_data = data if data is not None else self.preprocessed_data

        if input_data is None:
            raise ValueError("No data available for prediction")

        self.predictions = self.predictor.predict(input_data)
        return self.predictions

    def visualize(self):
        """
        Visualize model results
        """
        if not self.visualizer:
            raise NotImplementedError("Visualizer not initialized")

        if self.preprocessed_data is None:
            raise ValueError("No data available for visualization")

        self.visualizer.visualize(self.preprocessed_data, self.predictions)

    def evaluate(self, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            ground_truth (np.ndarray): Ground truth data

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.evaluator:
            raise NotImplementedError("Evaluator not initialized")

        if self.predictions is None:
            raise ValueError("No predictions available for evaluation")

        self.evaluation_results = self.evaluator.evaluate(
            ground_truth, self.predictions
        )
        return self.evaluation_results

    def run(self, data: np.ndarray, ground_truth: Optional[np.ndarray] = None):
        """
        Run the entire model pipeline

        Args:
            data (np.ndarray): Input data
            ground_truth (Optional[np.ndarray], optional): Ground truth for evaluation
        """
        self.preprocess(data)
        self.fit()
        self.predict()
        self.visualize()

        if ground_truth is not None:
            self.evaluate(ground_truth)
