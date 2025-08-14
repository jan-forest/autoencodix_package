import pytest
import numpy as np
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._datasplitter import DataSplitter


class TestDataSplitter:
    # FIXTURES --------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    @pytest.fixture
    def default_config(self):
        return DefaultConfig()

    @pytest.fixture
    def default_splitter(self, default_config):
        splitter = DataSplitter(config=default_config)
        return splitter

    @pytest.fixture
    def sample_data_size(self):
        return 100

    @pytest.fixture
    def edge_case_data_size(self):
        return 3

    # TESTS -----------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # basic split sizes -----------------------------------------------------------
    def test_train_split_has_correct_size(
        self, default_splitter, sample_data_size, default_config
    ):
        splits = default_splitter.split(sample_data_size)
        expected_len = int(default_config.train_ratio * sample_data_size)
        assert len(splits["train"]) == expected_len

    def test_validation_split_has_correct_size(
        self, default_config, sample_data_size, default_splitter
    ):
        splits = default_splitter.split(sample_data_size)
        expected_len = int(default_config.valid_ratio * sample_data_size)
        assert len(splits["valid"]) == expected_len

    def test_test_split_has_correct_size(
        self, default_config, sample_data_size, default_splitter
    ):
        splits = default_splitter.split(sample_data_size)
        expected_len = int(default_config.test_ratio * sample_data_size)
        assert len(splits["test"]) == expected_len

    # basic overlap checks --------------------------------------------------------
    def test_train_and_validation_splits_do_not_overlap(
        self, sample_data_size, default_splitter
    ):
        splits = default_splitter.split(sample_data_size)
        assert len(set(splits["train"]) & set(splits["valid"])) == 0

    def test_train_and_test_splits_do_not_overlap(self, sample_data_size, default_splitter):
        splits = default_splitter.split(sample_data_size)
        assert len(set(splits["train"]) & set(splits["test"])) == 0

    def test_validation_and_test_splits_do_not_overlap(
        self, sample_data_size, default_splitter
    ):
        splits = default_splitter.split(sample_data_size)
        assert len(set(splits["valid"]) & set(splits["test"])) == 0

    # input validation tests ------------------------------------------------------
    def test_ratio_validation(self, default_config):
        default_config.test_ratio = 0.4
        default_config.valid_ratio = 0.4
        default_config.train_ratio = 0.4
        with pytest.raises(ValueError) as exc_info:
            DataSplitter(default_config)
        assert "Split ratios must sum to 1" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_ratio", [-0.1, 1.1, 200, -100, -0.0001, 1.0001])
    def test_invalid_test_ratio_error_handling(self, default_config, invalid_ratio):
        default_config.test_ratio = invalid_ratio
        with pytest.raises(ValueError) as exc_info:
            DataSplitter(default_config)
        assert "Test ratio must be between 0 and 1" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_ratio", [-0.1, 1.1, 200, -100, -0.0001, 1.0001])
    def test_invalid_valid_ratio_error_handling(self, default_config, invalid_ratio):
        default_config.valid_ratio = invalid_ratio
        with pytest.raises(ValueError) as exc_info:
            DataSplitter(default_config)
        assert "Validation ratio must be between 0 and 1" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_ratio", [-0.1, 1.1, 200, -100, -0.0001, 1.0001])
    def test_invalid_train_ratio_error_handling(self, default_config, invalid_ratio):
        default_config.train_ratio = invalid_ratio
        with pytest.raises(ValueError) as exc_info:
            DataSplitter(default_config)
        assert "Train ratio must be between 0 and 1" in str(exc_info.value)

    def test_min_samples_validation(self, edge_case_data_size, default_splitter):
        # check if a ValueError is raised when the number of samples is less than the minimum required
        with pytest.raises(ValueError):
            default_splitter.split(edge_case_data_size)

    # custom split tests ----------------------------------------------------------
    @pytest.fixture
    def custom_splitter(self, default_config, sample_data_size):
        split = {
            "train": np.arange(int(sample_data_size * 0.6)),
            "valid": np.arange(
                int(sample_data_size * 0.6), int(sample_data_size * 0.70)
            ),
            "test": np.arange(int(sample_data_size * 0.70), sample_data_size),
        }
        splitter = DataSplitter(config=default_config, custom_splits=split)
        return splitter

    def test_custom_splitter_train_sizes(self, custom_splitter, sample_data_size):
        splits = custom_splitter.split(sample_data_size)
        assert len(splits["train"]) == int(sample_data_size * 0.6)

    def test_custom_splitter_valid_sizes(self, custom_splitter, sample_data_size):
        splits = custom_splitter.split(sample_data_size)
        assert len(splits["valid"]) == int(sample_data_size * 0.1)

    def test_custom_splitter_test_sizes(self, custom_splitter, sample_data_size):
        splits = custom_splitter.split(sample_data_size)
        assert len(splits["test"]) == int(sample_data_size * 0.3)

    @pytest.mark.parametrize(
        "invalid_split",
        [
            {
                "train:": np.arange(60),
                "valid": np.arange(50, 70),
                "test": np.arange(70, 100),
            },
            {
                "train": np.arange(60),
                "valid": np.arange(50, 70),
                "test": np.arange(50, 100),
            },
            {
                "train": np.arange(60),
                "valid": np.arange(50, 70),
                "test": np.arange(65, 70),
            },
        ],
    )
    def test_overlap_error_handling(self, default_config, invalid_split):
        with pytest.raises(ValueError):
            DataSplitter(default_config, custom_splits=invalid_split)

    def test_zero_train_split(self, sample_data_size):
        config = DefaultConfig()
        config.test_ratio = 0.5
        config.valid_ratio = 0.5
        config.train_ratio = 0.0
        splitter = DataSplitter(config=config)
        splits = splitter.split(sample_data_size)
        assert len(splits["train"]) == 0

    def test_zero_valid_split(self, sample_data_size):
        config = DefaultConfig()
        config.test_ratio = 0.5
        config.valid_ratio = 0.0
        config.train_ratio = 0.5
        splitter = DataSplitter(config=config)
        splits = splitter.split(sample_data_size)
        assert len(splits["valid"]) == 0

    def test_zero_test_split(self, sample_data_size):
        config = DefaultConfig()
        config.test_ratio = 0.0
        config.valid_ratio = 0.5
        config.train_ratio = 0.5
        splitter = DataSplitter(config=config)
        splits = splitter.split(sample_data_size)
        assert len(splits["test"]) == 0

    def test_zero_train_and_valid_split(self, sample_data_size):
        config = DefaultConfig()
        config.test_ratio = 1.0
        config.valid_ratio = 0.0
        config.train_ratio = 0.0
        splitter = DataSplitter(config=config)
        splits = splitter.split(sample_data_size)
        assert len(splits["train"]) == 0
        assert len(splits["valid"]) == 0

    def test_zero_train_and_test_split(self, sample_data_size):
        config = DefaultConfig()
        config.test_ratio = 0.0
        config.valid_ratio = 1.0
        config.train_ratio = 0.0
        splitter = DataSplitter(config=config)
        splits = splitter.split(sample_data_size)
        assert len(splits["train"]) == 0
        assert len(splits["test"]) == 0

    def test_zero_valid_and_test_split(self, sample_data_size):
        config = DefaultConfig()
        config.test_ratio = 0.0
        config.valid_ratio = 0.0
        config.train_ratio = 1.0
        splitter = DataSplitter(config=config)
        splits = splitter.split(sample_data_size)
        assert len(splits["valid"]) == 0
        assert len(splits["test"]) == 0

    def test_out_of_range_custom_indices(self, sample_data_size, default_config):
        custom_splits = {
            "train": np.append(np.arange(10), 1000),  # Out of range index
            "valid": np.arange(10, sample_data_size - 10),
            "test": np.arange(sample_data_size - 10, sample_data_size),
        }
        with pytest.raises(AssertionError):
            splitter = DataSplitter(default_config, custom_splits)
            _ = splitter.split(sample_data_size)