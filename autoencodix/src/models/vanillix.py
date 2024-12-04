from typing import Union, Optional, List

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from autoencodix.src.core._base_pipeline import BasePipeline
from autoencodix.src.core._default_config import DefaultConfig
from autoencodix.src.core._result import Result
from autoencodix.src.preprocessing import Preprocessor
from autoencodix.src.core._vanillix_architecture import VanillixArchitecture
from autoencodix.src.data._datasplitter import DataSplitter


class Trainer:
    """
    Handles model training process
    """
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Trainer
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Configuration dictionary for training
        """
        self.config = config or {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-3
        }
        self._train_losses = []
        self._valid_losses = []
    
    def prepare_dataloaders(
        self, 
        datasets: Dict[str, CustomDataset]
    ) -> Dict[str, DataLoader]:
        """
        Create dataloaders from datasets
        
        Parameters:
        -----------
        datasets : Dict[str, CustomDataset]
            Datasets for each split
        
        Returns:
        --------
        Dict[str, DataLoader]
            Dataloaders for each split
        """
        return {
            split: DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=(split == 'train')
            )
            for split, dataset in datasets.items()
        }
    
    def train(
        self, 
        model: torch.nn.Module, 
        dataloaders: Dict[str, DataLoader]
    ) -> torch.nn.Module:
        """
        Train the model
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        dataloaders : Dict[str, DataLoader]
            Dataloaders for each split
        
        Returns:
        --------
        torch.nn.Module
            Trained model
        """
        # Optimization setup
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate']
        )
        criterion = torch.nn.MSELoss()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch in dataloaders['train']:
                # Assumes batch is either features or (features, labels)
                inputs = batch[0] if isinstance(batch, tuple) else batch
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, inputs)  # Autoencoder reconstruction loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in dataloaders['valid']:
                    inputs = batch[0] if isinstance(batch, tuple) else batch
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    valid_loss += loss.item()
            
            # Record losses
            self._train_losses.append(train_loss / len(dataloaders['train']))
            self._valid_losses.append(valid_loss / len(dataloaders['valid']))
        
        return model

class Predictor:
    """
    Handles prediction process
    """
    def __init__(
        self, 
        model: torch.nn.Module,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Predictor
        
        Parameters:
        -----------
        model : torch.nn.Module
            Trained model for prediction
        config : Optional[Dict[str, Any]]
            Configuration dictionary for prediction
        """
        self.model = model
        self.config = config or {
            'batch_size': 32
        }
    
    def predict(
        self, 
        dataset: CustomDataset
    ) -> np.ndarray:
        """
        Make predictions on a dataset
        
        Parameters:
        -----------
        dataset : CustomDataset
            Dataset to make predictions on
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect predictions
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle both labeled and unlabeled datasets
                inputs = batch[0] if isinstance(batch, tuple) else batch
                
                # Get model outputs
                outputs = self.model(inputs)
                
                # Convert to numpy and append
                predictions.append(outputs.numpy())
        
        # Concatenate and return predictions
        return np.concatenate(predictions)
@dataclass
class DataSetContainer:
    """
    A dataclass to store datasets for training, validation, and testing.
    Attributes
    ----------
    train : CustomDataset
        The training dataset.
    valid : CustomDataset
        The validation dataset.
    test : CustomDataset
        The testing dataset.
    """
    train: CustomDataset
    valid: CustomDataset
    test: CustomDataset

class Vanillix(BasePipeline):
    def __init__(
        self,
        data: Union[np.ndarray, AnnData, pd.DataFrame],
        config: Optional[DefaultConfig] = None,
        data_splitter: Optional[DataSplitter] = None
    ):
        super().__init__(data, config)
        self.data = data
        self.x = None
        self.model = VanillixArchitecture(config=self.config)
        self.preprocessor = Preprocessor(config=self.config)
        self.trainer = Trainer(config=self.config)
        self.config = config
        self.result = Result()

        self._datasets = {}
        self._is_fitted = False
        if data_splitter is None:
            self.data_splitter = DataSplitter(config=self.config)


    def _build_datasets(self):
        """
        Takes the self.x attribute and creates train, valid, and test datasets.
        If no Datasplitter is provided, it will split the data according to the default configuration.

        
        """
        self._datasets = self.data_splitter.split(self.x)
        pass
        

    def preprocess(self) -> None:
        """
        Takes the user input data and filters, norrmalizes and cleans the data.
        Populates the self.x attribute with the preprocessed data as a numpy array.

        """
        self.x = self.preprocessor.preprocess(self.data)


    def fit(
        self, 
    ) -> None:
        """
        Trains the model using the training data and updates the model attribute.
        """
        if self.x is None:
            self.preprocess()
        
        self._build_datasets()
        self.model = self.trainer.train(self.model, self._datasets)
        # populate result with mu, logvar, reconsturction and losses

        self._is_fitted = True

    def predict(
        self, 
        data: Optional[Union[np.ndarray, pd.DataFrame, AnnData]] = None
    ) -> np.ndarray:
        """
        Prediction method with flexibility
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # If no data is provided, use test dataset
        if data is None:
            # Explicitly check for test dataset
            if 'test' not in self._datasets:
                raise ValueError(
                    "No test dataset available. "
                    "Ensure a test split was created during fitting, or pass new data for prediction."
                )
            # should return mu
            prediction = self.predictor.predict(self._datasets.test)
            
        # test if data has correct type
        elif not isinstance(data, (np.ndarray, pd.DataFrame, AnnData)):
            raise TypeError(
                f"Expected data type to be one of np.ndarray, AnnData, or pd.DataFrame, got {type(data)}."
            )
        else:
            # Preprocess and create dataset for new data
            processed_data = self.preprocessor.preprocess(data)
        