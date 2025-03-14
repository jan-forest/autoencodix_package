import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from torch.utils.data import Dataset
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.data._datasetcontainer import DatasetContainer


class TestBasePipeline:
    @pytest.fixture
    def mock_dataset_container(self):
        # Create a mock DatasetContainer with train, valid, and test datasets
        mock_train = MagicMock(spec=Dataset)
        mock_valid = MagicMock(spec=Dataset)
        mock_test = MagicMock(spec=Dataset)
        container = MagicMock(spec=DatasetContainer)
        container.train = mock_train
        container.valid = mock_valid
        container.test = mock_test
        return container

    def test_initialization_with_valid_data(self):
        valid_data_types = [
            np.random.rand(100, 10),
            pd.DataFrame(np.random.rand(100, 10)),
        ]
        for data in valid_data_types:
            pipeline = BasePipeline(
                processed_data=data,
                dataset_type=Mock(),
                trainer_type=Mock(),
                loss_type=Mock(),
                model_type=Mock(),
                datasplitter_type=Mock,
                preprocessor_type=Mock(),
                visualizer=Mock(),
                evaluator=Mock(),
                result=Mock(),
                custom_split=None,
            )
            assert pipeline is not None

    @pytest.mark.parametrize(
        "invalid_data", [[1, 2, 3], "string data", {"dict": "data"}]
    )
    def test_initialization_raises_error_for_invalid_data(self, invalid_data):
        with pytest.raises(TypeError):
            BasePipeline(
                processed_data=invalid_data,
                dataset_type=Mock(),
                trainer_type=Mock(),
                loss_type=Mock(),
                model_type=Mock(),
                datasplitter_type=Mock,
                preprocessor_type=Mock(),
                visualizer=Mock(),
                evaluator=Mock(),
                result=Mock(),
                custom_split=None,
            )

    def test_preprocess_raises_error_without_preprocessor(self):
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=None,  # This will cause the error
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        with pytest.raises(NotImplementedError):
            pipeline.preprocess()

    def test_fit_without_preprocessed_data_raises_error(self):
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            loss_type=Mock(),
            model_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        # _datasets is None by default
        with pytest.raises(ValueError):
            pipeline.fit()

    def test_predict_without_fitted_model_raises_error(self, mock_dataset_container):
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        # Preprocessor not initialized
        with pytest.raises(NotImplementedError):
            pipeline.predict(data=mock_dataset_container)

    def test_predict_without_preprocessed_data_raises_error(self, mock_dataset_container):
        # Set up pipeline with initialized preprocessor but no datasets
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        pipeline._preprocessor = Mock()  # Manually set preprocessor
        
        with pytest.raises(NotImplementedError):
            pipeline.predict(data=mock_dataset_container)

    def test_predict_without_trained_model_raises_error(self, mock_dataset_container):
        # Set up pipeline with preprocessor and datasets but no trained model
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        pipeline._preprocessor = Mock()
        pipeline._datasets = mock_dataset_container
        pipeline.result.model = None  # Ensure model is None
        
        with pytest.raises(NotImplementedError):
            pipeline.predict(data=mock_dataset_container)

    def test_predict_with_no_test_data_raises_error(self, mock_dataset_container):
        # Set up pipeline with preprocessor, datasets, and trained model but empty test dataset
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        pipeline._preprocessor = Mock()
        pipeline._datasets = mock_dataset_container
        pipeline._datasets.test = None  # Remove test dataset
        pipeline.result.model = Mock()  # Add mock model
        
        # Create an empty dataset container for prediction
        empty_container = MagicMock(spec=DatasetContainer)
        empty_container.test = None
        
        with pytest.raises(ValueError):
            pipeline.predict(data=empty_container)

    def test_visualize_without_visualizer_raises_error(self):
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=None,  # No visualizer
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        
        with pytest.raises(NotImplementedError):
            pipeline.visualize()
            
    def test_validate_container_with_invalid_datasets(self):
        pipeline = BasePipeline(
            processed_data=np.random.rand(100, 10),  # This will be overwritten
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            evaluator=Mock(),
            result=Mock(),
            custom_split=None,
        )
        
        # Test with invalid train dataset (not None and not a Dataset)
        invalid_container = MagicMock(spec=DatasetContainer)
        invalid_container.train = "not a dataset"
        invalid_container.valid = None
        invalid_container.test = None
        
        pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            pipeline._validate_container()
            
        # Test with invalid valid dataset
        invalid_container.train = MagicMock(spec=Dataset)
        invalid_container.valid = "not a dataset"
        
        pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            pipeline._validate_container()
            
        # Test with invalid test dataset
        invalid_container.valid = None
        invalid_container.test = "not a dataset"
        
        pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            pipeline._validate_container()
            
        # Test with all None datasets
        invalid_container.train = None
        invalid_container.test = None
        
        pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            pipeline._validate_container()