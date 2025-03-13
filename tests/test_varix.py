import numpy as np
import pandas as pd
import pytest

from autoencodix.utils.default_config import DefaultConfig
from autoencodix.varix import Varix


@pytest.fixture
def sample_data():
    return np.zeros((100, 10))


@pytest.fixture
def large_sample_data():
    return np.ones((500, 20))


class TestVarix:
    def test_varix_initialization(self, sample_data):
        varix = Varix(preprocessed_data=sample_data)
        assert varix.data is not None
        assert varix.config is not None
        assert varix._preprocessor is not None

    def test_varix_custom_configuration(self, sample_data):
        custom_config = DefaultConfig(
            epochs=1, checkpoint_interval=1, device="cpu", batch_size=1
        )

        varix = Varix(preprocessed_data=sample_data, config=custom_config)
        assert varix.config.batch_size == 1

    def test_full_pipeline_workflow(self, large_sample_data):
        varix = Varix(preprocessed_data=large_sample_data, config=DefaultConfig(epochs=1))
        result = varix.run()

        assert result.preprocessed_data is not None
        assert result.datasets is not None
        assert result.model is not None

    @pytest.mark.parametrize(
        "data", [np.zeros((100, 10)), pd.DataFrame(np.zeros((100, 10)))]
    )
    def test_pipeline_with_different_data_types(self, data):
        # skip for now since pandas DataFrame is not implemented
        if isinstance(data, pd.DataFrame):
            pytest.skip("Pandas DataFrame not implemented")
        else:
            varix = Varix(preprocessed_data=data, config=DefaultConfig(epochs=1))
            result = varix.run()

        assert result is not None

    def test_prediction_with_new_data(self, sample_data):
        varix = Varix(preprocessed_data=sample_data)
        result = varix.run()
        old_reconstruction = result.reconstructions.get(split="test", epoch=-1)

        new_data = np.zeros((10, 10))
        varix.predict(new_data)
        predict_result = varix.result
        new_reconstruction = predict_result.reconstructions.get(split="test", epoch=-1)
        print(f"new_data shape: {new_data.shape}")
        print(f"new reconstruction shape: {new_reconstruction.shape}")
        assert not np.array_equal(old_reconstruction, new_reconstruction)
