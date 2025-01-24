import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from autoencodix.base._base_pipeline import BasePipeline


class TestBasePipeline:
    def test_initialization_with_valid_data(self):
        valid_data_types = [
            np.random.rand(100, 10),
            pd.DataFrame(np.random.rand(100, 10)),
        ]

        for data in valid_data_types:
            pipeline = BasePipeline(
                data=data,
                dataset_type=Mock(),
                trainer_type=Mock(),
                model_type=Mock(),
                datasplitter_type=Mock,
                preprocessor=Mock(),
                predictor=Mock(),
                visualizer=Mock(),
                result=Mock(),
            )
            assert pipeline is not None

    @pytest.mark.parametrize(
        "invalid_data", [[1, 2, 3], "string data", {"dict": "data"}]
    )
    def test_initialization_raises_error_for_invalid_data(self, invalid_data):
        with pytest.raises(TypeError):
            BasePipeline(
                data=invalid_data,
                dataset_type=Mock(),
                trainer_type=Mock(),
                model_type=Mock(),
                datasplitter_type=Mock,
                preprocessor=Mock(),
                predictor=Mock(),
                visualizer=Mock(),
                result=Mock(),
            )

    def test_preprocess_raises_error_without_preprocessor(self):
        pipeline = BasePipeline(
            data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            datasplitter_type=Mock,
            preprocessor=None,
            predictor=Mock(),
            visualizer=Mock(),
            result=Mock(),
        )

        with pytest.raises(NotImplementedError):
            pipeline.preprocess()

    def test_fit_without_preprocessed_data_raises_error(self):
        pipeline = BasePipeline(
            data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            datasplitter_type=Mock,
            preprocessor=Mock(),
            predictor=Mock(),
            visualizer=Mock(),
            result=Mock(),
        )

        with pytest.raises(ValueError):
            pipeline.fit()

    def test_predict_without_fitted_model_raises_error(self):
        pipeline = BasePipeline(
            data=np.random.rand(100, 10),
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            datasplitter_type=Mock,
            preprocessor=Mock(),
            predictor=Mock(),
            visualizer=Mock(),
            result=Mock(),
        )
        with pytest.raises(NotImplementedError):
            pipeline.predict()
