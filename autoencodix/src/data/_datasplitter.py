
import numpy as np
from typing import Dict, List, Union, Optional
from autoencodix.src.data._numeric_dataset import NumericDataset
from autoencodix.src.core._default_config import DefaultConfig

class DataSplitter:
    """
    Receives the processed data as np.ndarray and splits it into train, validation, and test sets.
    Uses the default train, test and valid ratio form the default config. If the user overwrites the default config,
    these ratios will be used. The user can also provide custom indices for each split.
    """
    def __init__(
        self, 
        config: DefaultConfig
    ):
        """
        Initialize DataSplitter
        
        Parameters:
        -----------
        test_size : float, optional
            Proportion of data to use for testing
        valid_size : float, optional
            Proportion of data to use for validation
        random_state : int, optional
            Random seed for reproducibility
        """
        self.test_size = test_size
        self.valid_size = valid_size
        self.random_state = random_state
    
    def split(
        self, 
        X: np.ndarray, 
        custom_splits: Optional[Dict[str, Union[List[int], np.ndarray]]] = None,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, NumericDataset]:
        """
        Create train, validation, and test datasets
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        custom_splits : Optional[Dict[str, Union[List[int], np.ndarray]]]
            Optional dictionary of custom indices for each split
        y : Optional[np.ndarray]
            Optional labels for supervised learning
        
        Returns:
        --------
        Dict[str, CustomDataset]
            Datasets for each split
        """
        # Use custom splits if provided
        if custom_splits:
            return self._split_with_custom_indices(X, custom_splits, y)
        
        # Perform random splitting
        from sklearn.model_selection import train_test_split
        
        # First split into train+valid and test
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X, 
            y if y is not None else np.zeros(len(X)),
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Split train+valid into train and valid
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, 
            y_train_valid,
            test_size=self.valid_size, 
            random_state=self.random_state
        )
        
        # Create datasets
        return {
            'train': NumericDataset(X_train, y_train),
            'valid': NumericDataset(X_valid, y_valid),
            'test': NumericDataset(X_test, y_test)
        }
    
    def _split_with_custom_indices(
        self, 
        X: np.ndarray, 
        custom_splits: Dict[str, Union[List[int], np.ndarray]],
        y: Optional[np.ndarray] = None
    ) -> Dict[str, NumericDataset]:
        """
        Split data using custom indices
        
        Parameters:
        -----------
        X : np.ndarray
            Full input features
        custom_splits : Dict[str, Union[List[int], np.ndarray]]
            Dictionary of indices for each split
        y : Optional[np.ndarray]
            Optional labels
        
        Returns:
        --------
        Dict[str, CustomDataset]
            Datasets for each split
        """
        split_datasets = {}
        
        for split_name, indices in custom_splits.items():
            # Convert indices to numpy array if they're not already
            indices = np.array(indices)
            
            # Select data for this split
            split_data = X[indices]
            
            # Select corresponding labels if provided
            split_labels = y[indices] if y is not None else None
            
            # Convert to custom dataset
            split_datasets[split_name] = NumericDataset(split_data, split_labels)
        
        return split_datasets