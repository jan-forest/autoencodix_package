from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
from scipy.sparse import issparse

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.utils.default_config import DefaultConfig


class StackixPreprocessor(BasePreprocessor):
    """
    Preprocessor for Stackix architecture, which handles multiple modalities separately.

    Unlike GeneralPreprocessor which combines all modalities, StackixPreprocessor
    keeps modalities separate for individual VAE training in the Stackix architecture.
    
    Attributes
    ----------
    config : DefaultConfig
        Configuration parameters for preprocessing and model architecture
    _datapackage : Optional[Dict[str, Any]]
        Dictionary storing processed data splits
    _dataset_container : DatasetContainer
        Container for processed datasets by split
    """

    def __init__(self, config: DefaultConfig):
        """
        Initialize the StackixPreprocessor with the given configuration.
        
        Parameters
        ----------
        config : DefaultConfig
            Configuration parameters for preprocessing
        """
        super().__init__(config=config)
        self._datapackage: Optional[Dict[str, Any]] = None
        self._dataset_container: Optional[DatasetContainer] = None

    def _extract_primary_data(self, modality_data: Any) -> np.ndarray:
        """
        Extract primary data matrix and convert to dense array if sparse.
        
        Parameters
        ----------
        modality_data : Any
            Input modality data with X attribute
            
        Returns
        -------
        np.ndarray
            Extracted data as dense numpy array
        """
        primary_data = modality_data.X
        if issparse(primary_data):
            primary_data = primary_data.toarray()
        return primary_data

    def _combine_layers(
        self, modality_name: str, modality_data: Any
    ) -> List[np.ndarray]:
        """
        Combine specified layers from a modality.
        
        Parameters
        ----------
        modality_name : str
            Name of the modality
        modality_data : Any
            Input modality data with layers attribute
            
        Returns
        -------
        List[np.ndarray]
            List of combined layer data as numpy arrays
        """
        layer_list: List[np.ndarray] = []
        
        # Check if data_info and selected_layers are properly configured
        if (not hasattr(self.config, 'data_config') or 
            not hasattr(self.config.data_config, 'data_info') or
            modality_name not in self.config.data_config.data_info or
            not hasattr(self.config.data_config.data_info[modality_name], 'selected_layers')):
            # Default to using just the primary data if configuration is missing
            layer_list.append(self._extract_primary_data(modality_data))
            return layer_list
            
        selected_layers = self.config.data_config.data_info[
            modality_name
        ].selected_layers

        for layer_name in selected_layers:
            if layer_name == "X":
                primary_data = self._extract_primary_data(modality_data)
                layer_list.append(primary_data)
            elif hasattr(modality_data, 'layers') and layer_name in modality_data.layers:
                layer_data = modality_data.layers[layer_name]
                if issparse(layer_data):
                    layer_data = layer_data.toarray()
                layer_list.append(layer_data)
            else:
                print(
                    f"Layer '{layer_name}' not found in modality '{modality_name}'. Skipping."
                )

        return layer_list

    def _process_multi_bulk(self, data_dict: Dict[str, Any]) -> StackixDataset:
        """
        Process multi-bulk data for Stackix architecture.
        
        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing data and indices
            
        Returns
        -------
        StackixDataset
            Processed dataset with multiple modalities
            
        Raises
        ------
        ValueError
            If no multi_bulk data is found
        """
        data, split_indices = data_dict["data"], data_dict.get("indices")
        
        if not hasattr(data, 'multi_bulk') or data.multi_bulk is None:
            raise ValueError("No multi_bulk data found")
            
        multi_bulk = data.multi_bulk

        # Create dictionary to hold datasets for each modality
        modality_datasets: Dict[str, BaseDataset] = {}
        
        # Process each modality separately
        for modality_name, df in multi_bulk.items():
            if df is None:
                continue
                
            # Apply any preprocessing transformations if configured
            processed_df = self._apply_transformations(df, modality_name)
            
            # Convert to tensor
            tensor_data = torch.from_numpy(processed_df.values).float()
            
            # Create a NumericDataset for this modality
            modality_dataset = NumericDataset(
                data=tensor_data,
                config=self.config,
                sample_ids=processed_df.index.tolist(),
                feature_ids=processed_df.columns.tolist()
            )
            
            modality_datasets[modality_name] = modality_dataset

        # Create StackixDataset from the modality datasets
        return StackixDataset.from_datasets(datasets=modality_datasets, config=self.config)

    def _apply_transformations(self, df, modality_name: str):
        """
        Apply configured transformations to dataframe.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe to transform
        modality_name : str
            Name of the modality for configuration lookup
            
        Returns
        -------
        pandas.DataFrame
            Transformed dataframe
        """
        # Check if transformations are configured for this modality
        if (hasattr(self.config, 'data_config') and 
            hasattr(self.config.data_config, 'transformations') and
            modality_name in self.config.data_config.transformations):
            
            transformations = self.config.data_config.transformations[modality_name]
            for transform in transformations:
                if hasattr(transform, 'fit_transform'):
                    df = transform.fit_transform(df)
                elif hasattr(transform, '__call__'):
                    df = transform(df)
        
        return df

    def _process_multi_sc(self, data_dict: Dict[str, Any]) -> StackixDataset:
        """
        Process multi-single-cell data for Stackix architecture.
        
        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing multi-single-cell data
            
        Returns
        -------
        StackixDataset
            Processed dataset with multiple modalities
            
        Raises
        ------
        ValueError
            If no multi_sc data is found
        """
        data = data_dict["data"]
        
        if not hasattr(data, 'multi_sc') or data.multi_sc is None:
            raise ValueError("No multi_sc data found")
            
        multi_sc = data.multi_sc

        # Create dictionary to hold datasets for each modality
        modality_datasets: Dict[str, BaseDataset] = {}

        # Process each modality separately
        if hasattr(multi_sc, 'mod'):
            for modality_name, modality_data in multi_sc.mod.items():
                # Combine layers for this modality
                combined_layers = self._combine_layers(
                    modality_name=modality_name, modality_data=modality_data
                )

                if combined_layers:
                    # Concatenate along feature axis if multiple layers
                    if len(combined_layers) > 1:
                        combined_data = np.concatenate(combined_layers, axis=1)
                    else:
                        combined_data = combined_layers[0]

                    # Apply any additional preprocessing
                    combined_data = self._apply_sc_transformations(combined_data, modality_name)
                    
                    # Convert to tensor
                    tensor_data = torch.from_numpy(combined_data).float()
                    
                    # Get sample and feature IDs if available
                    sample_ids = modality_data.obs_names.tolist() if hasattr(modality_data, 'obs_names') else None
                    feature_ids = modality_data.var_names.tolist() if hasattr(modality_data, 'var_names') else None
                    
                    # Create a NumericDataset for this modality
                    modality_dataset = NumericDataset(
                        data=tensor_data,
                        config=self.config,
                        sample_ids=sample_ids,
                        feature_ids=feature_ids
                    )
                    
                    modality_datasets[modality_name] = modality_dataset

        # Create StackixDataset from the modality datasets
        return StackixDataset.from_datasets(datasets=modality_datasets, config=self.config)
        
    def _apply_sc_transformations(self, data: np.ndarray, modality_name: str) -> np.ndarray:
        """
        Apply configured transformations to single-cell data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data to transform
        modality_name : str
            Name of the modality for configuration lookup
            
        Returns
        -------
        np.ndarray
            Transformed data
        """
        # Apply any single-cell specific transformations configured for this modality
        if (hasattr(self.config, 'data_config') and 
            hasattr(self.config.data_config, 'sc_transformations') and
            modality_name in self.config.data_config.sc_transformations):
            
            transformations = self.config.data_config.sc_transformations[modality_name]
            for transform in transformations:
                if hasattr(transform, 'fit_transform'):
                    data = transform.fit_transform(data)
                elif hasattr(transform, '__call__'):
                    data = transform(data)
        
        return data

    def _process_data_package(self, data_dict: Dict[str, Any]) -> BaseDataset:
        """
        Process a data package based on its type for Stackix architecture.
        
        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing data to process
            
        Returns
        -------
        BaseDataset
            Processed dataset
            
        Raises
        ------
        ValueError
            If no supported data type is found
        """
        data = data_dict["data"]

        # First try to process as multi_bulk
        try:
            if hasattr(data, 'multi_bulk') and data.multi_bulk is not None:
                return self._process_multi_bulk(data_dict)
        except Exception as e:
            print(f"Error processing multi_bulk: {e}")
        
        # Then try to process as multi_sc
        try:
            if hasattr(data, 'multi_sc') and data.multi_sc is not None:
                return self._process_multi_sc(data_dict)
        except Exception as e:
            print(f"Error processing multi_sc: {e}")
            
        # If we made it here, check annotations as fallback
        if hasattr(data, '__annotations__'):
            for key in data.__annotations__.keys():
                attr_val = getattr(data, key, None)
                if key == "multi_bulk" and attr_val is not None:
                    return self._process_multi_bulk(data_dict)
                elif key == "multi_sc" and attr_val is not None:
                    return self._process_multi_sc(data_dict)

        raise ValueError("No supported data type found in the split")

    def preprocess(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> DatasetContainer:
        """
        Execute preprocessing steps for Stackix architecture.

        Unlike GeneralPreprocessor, this keeps modalities separate for individual VAE training.
        
        Parameters
        ----------
        raw_user_data : Optional[DataPackage]
            Raw user data to preprocess, or None to use self._datapackage
            
        Returns
        -------
        DatasetContainer
            Container with StackixDataset for each split
            
        Raises
        ------
        TypeError
            If datapackage is None after preprocessing
        """
        self._datapackage = self._general_preprocess(raw_user_data)
        if self._datapackage is None:
            raise TypeError("Datapackage cannot be None")

        self._dataset_container = DatasetContainer()

        for split in ["train", "valid", "test"]:
            if split not in self._datapackage or self._datapackage[split].get("data") is None:
                self._dataset_container[split] = None
                continue
                
            try:
                dataset = self._process_data_package(data_dict=self._datapackage[split])
                self._dataset_container[split] = dataset
            except Exception as e:
                print(f"Error processing {split} split: {e}")
                self._dataset_container[split] = None

        return self._dataset_container
        
    def get_modality_datasets(self) -> Dict[str, Dict[str, BaseDataset]]:
        """
        Get all modality datasets organized by split.
        
        Returns
        -------
        Dict[str, Dict[str, BaseDataset]]
            Dictionary mapping splits to dictionaries of modality datasets
            
        Raises
        ------
        ValueError
            If no datasets are available
        """
        if not self._dataset_container:
            raise ValueError("No datasets available")
            
        result = {}
        
        for split_name in ["train", "valid", "test"]:
            split_dataset = getattr(self._dataset_container, split_name)
            
            if split_dataset and isinstance(split_dataset, StackixDataset):
                # Extract modality datasets from each split
                datasets_dict = {}
                
                for modality_name, modality_dataset in split_dataset.datasets_dict.items():
                    datasets_dict[modality_name] = modality_dataset
                
                if datasets_dict:
                    result[split_name] = datasets_dict
        
        if not result:
            raise ValueError("No modality datasets found")
            
        return result
        
    def get_modality_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of each modality in the training dataset.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping modality names to their input dimensions
            
        Raises
        ------
        ValueError
            If no training dataset is available or it's not a StackixDataset
        """
        if not self._dataset_container or not self._dataset_container.train:
            raise ValueError("No training dataset available")
            
        train_ds = self._dataset_container.train
        if not isinstance(train_ds, StackixDataset):
            raise ValueError("Training dataset is not a StackixDataset")
            
        return {
            modality: dataset.get_input_dim() 
            for modality, dataset in train_ds.datasets_dict.items()
        }