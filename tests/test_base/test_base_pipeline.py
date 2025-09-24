from unittest.mock import MagicMock, Mock

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import pytest
from mudata import MuData  # type: ignore
from torch.utils.data import Dataset

from autoencodix.vanillix import Vanillix
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.default_config import DataCase, DataConfig, DefaultConfig

"""
We use Vanillix, because we cannot use BasePipeline due to abstraction.
We test only BasePipeline specifics, not Vanillix specifics here.
"""


class MockDefaultConfig(DefaultConfig):
    pass


class TestBasePipeline:
    @pytest.fixture
    def mock_dataset_container(self):
        container = MagicMock(spec=DatasetContainer)
        container.train = MagicMock(spec=Dataset)
        container.valid = MagicMock(spec=Dataset)
        container.test = MagicMock(spec=Dataset)
        return container

    @pytest.fixture
    def valid_config(self):
        config = MockDefaultConfig()
        config.data_case = MagicMock(spec=DataCase)
        config.data_config = MagicMock(spec=DataConfig)
        config.data_config.data_info = {"test": MagicMock()}
        return config

    @pytest.fixture
    def minimal_pipeline(self, valid_config):
        bp = Vanillix(
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            data=None,
            evaluator=Mock(),
            result=Mock(),
            config=valid_config,
        )
        return bp

    def test_initialization_with_valid_data_infered_data_case(
        self, mock_dataset_container
    ):
        valid_data_types = [
            mock_dataset_container,
            ad.AnnData(np.random.rand(10, 5)),
            MuData({"mod": ad.AnnData(np.random.rand(10, 5))}),
            pd.DataFrame(np.random.rand(10, 5)),
            {"key": pd.DataFrame(np.random.rand(10, 5))},
        ]

        for data in valid_data_types:
            pipeline = Vanillix(
                dataset_type=Mock(),
                trainer_type=Mock(),
                model_type=Mock(),
                loss_type=Mock(),
                datasplitter_type=Mock,
                preprocessor_type=Mock(),
                visualizer=Mock(),
                data=data,
                evaluator=Mock(),
                result=Mock(),
            )
            assert (
                pipeline.raw_user_data is not None
                or pipeline.preprocessed_data is not None
            )

    def test_initialization_with_valid_data_package(self, valid_config):
        pipeline = Vanillix(
            dataset_type=Mock(),
            config=valid_config,
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            data=DataPackage(
                multi_bulk={"multi_bulk:": pd.DataFrame(np.random.rand(10, 5))}
            ),
            evaluator=Mock(),
            result=Mock(),
        )
        assert (
            pipeline.raw_user_data is not None or pipeline.preprocessed_data is not None
        )

    @pytest.mark.parametrize("invalid_data", [[1, 2, 3], "string data", 123])
    def test_initialization_raises_error_for_invalid_data(self, invalid_data):
        with pytest.raises(TypeError):
            Vanillix(
                dataset_type=Mock(),
                trainer_type=Mock(),
                model_type=Mock(),
                loss_type=Mock(),
                datasplitter_type=Mock,
                preprocessor_type=Mock(),
                visualizer=Mock(),
                data=invalid_data,
                evaluator=Mock(),
                result=Mock(),
            )

    def test_preprocess_raises_error_without_preprocessor(self, minimal_pipeline):
        minimal_pipeline._preprocessor_type = None
        with pytest.raises(NotImplementedError):
            minimal_pipeline.preprocess()

    def test_fit_without_preprocessed_data_raises_error(self, minimal_pipeline):
        minimal_pipeline._datasets = None
        with pytest.raises(ValueError):
            minimal_pipeline.fit()

    def test_predict_without_fitted_model_raises_error(
        self, minimal_pipeline, mock_dataset_container
    ):
        minimal_pipeline.result.model = None
        with pytest.raises(NotImplementedError):
            minimal_pipeline.predict(data=mock_dataset_container)

    def test_predict_without_preprocessor_raises_error(
        self, minimal_pipeline, mock_dataset_container
    ):
        minimal_pipeline._preprocessor = None
        with pytest.raises(NotImplementedError):
            minimal_pipeline.predict(data=mock_dataset_container)

    def test_predict_with_no_test_data_raises_error(
        self, minimal_pipeline, mock_dataset_container
    ):
        mock_dataset_container.test = None
        with pytest.raises(ValueError):
            minimal_pipeline.predict(data=mock_dataset_container)

    def test_visualize_without_visualizer_raises_error(self, minimal_pipeline):
        minimal_pipeline._visualizer = None
        with pytest.raises(NotImplementedError):
            minimal_pipeline.visualize()

    def test_validate_container_with_invalid_train_dataset(self, minimal_pipeline):
        invalid_container = MagicMock(spec=DatasetContainer)
        invalid_container.train = "invalid_type"
        invalid_container.valid = MagicMock(spec=Dataset)
        invalid_container.test = MagicMock(spec=Dataset)

        minimal_pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            minimal_pipeline._validate_container()

    def test_validate_container_with_invalid_valid_dataset(self, minimal_pipeline):
        invalid_container = MagicMock(spec=DatasetContainer)
        invalid_container.train = MagicMock(spec=Dataset)
        invalid_container.valid = "invalid_type"
        invalid_container.test = MagicMock(spec=Dataset)

        minimal_pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            minimal_pipeline._validate_container()

    def test_validate_container_with_invalid_test_dataset(self, minimal_pipeline):
        invalid_container = MagicMock(spec=DatasetContainer)
        invalid_container.train = MagicMock(spec=Dataset)
        invalid_container.valid = MagicMock(spec=Dataset)
        invalid_container.test = "invalid_type"

        minimal_pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            minimal_pipeline._validate_container()

    def test_validate_container_with_all_none_datasets(self, minimal_pipeline):
        invalid_container = MagicMock(spec=DatasetContainer)
        invalid_container.train = None
        invalid_container.valid = None
        invalid_container.test = None

        minimal_pipeline.preprocessed_data = invalid_container
        with pytest.raises(ValueError):
            minimal_pipeline._validate_container()

    def test_handle_direct_user_data_with_anndata(self):
        pipeline = Vanillix(
            dataset_type=Mock(),
            trainer_type=Mock(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            data=ad.AnnData(np.random.rand(10, 5)),
            evaluator=Mock(),
            result=Mock(),
        )
        assert isinstance(pipeline.raw_user_data, DataPackage)
        assert pipeline.config.data_case == DataCase.MULTI_SINGLE_CELL
        del pipeline

    def test_handle_direct_user_data_with_dataframe(self):
        pipeline = Vanillix(
            dataset_type=Mock(),
            trainer_type=Mock(),
            config=MockDefaultConfig(),
            model_type=Mock(),
            loss_type=Mock(),
            datasplitter_type=Mock,
            preprocessor_type=Mock(),
            visualizer=Mock(),
            data=pd.DataFrame(np.random.rand(10, 5)),
            evaluator=Mock(),
            result=Mock(),
        )
        assert isinstance(pipeline.raw_user_data, DataPackage)
        print(f"Data case: {pipeline.config.data_case}")
        assert pipeline.config.data_case == DataCase.MULTI_BULK

    def test_run_method_returns_result(self, minimal_pipeline):
        minimal_pipeline.preprocess = MagicMock()
        minimal_pipeline.fit = MagicMock()
        minimal_pipeline.predict = MagicMock()
        minimal_pipeline.evaluate = MagicMock()
        minimal_pipeline.visualize = MagicMock()

        result = minimal_pipeline.run()
        assert result is not None
