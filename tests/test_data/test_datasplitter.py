import pytest
import numpy as np
import pandas as pd
from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._datasplitter import DataSplitter, PairedUnpairedSplitter


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

    def test_train_and_test_splits_do_not_overlap(
        self, sample_data_size, default_splitter
    ):
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


# Fixture for sample data
@pytest.fixture
def sample_groups():
    return {
        "rna_only": [f"rna_only{i}" for i in range(1, 11)],  # 10 samples
        "protein_only": [f"protein_only{i}" for i in range(1, 11)],
        "meth_only": [f"meth_only{i}" for i in range(1, 11)],
        "rna_protein": [f"rna_protein{i}" for i in range(1, 11)],
        "rna_meth": [f"rna_meth{i}" for i in range(1, 11)],
        "protein_meth": [f"protein_meth{i}" for i in range(1, 11)],
        "rna_protein_meth": [f"rna_protein_meth{i}" for i in range(1, 11)],
    }


@pytest.fixture
def data_package(sample_groups):
    rna_samples = (
        sample_groups["rna_only"]
        + sample_groups["rna_protein"]
        + sample_groups["rna_meth"]
        + sample_groups["rna_protein_meth"]
    )
    protein_samples = (
        sample_groups["protein_only"]
        + sample_groups["rna_protein"]
        + sample_groups["protein_meth"]
        + sample_groups["rna_protein_meth"]
    )
    meth_samples = (
        sample_groups["meth_only"]
        + sample_groups["rna_meth"]
        + sample_groups["protein_meth"]
        + sample_groups["rna_protein_meth"]
    )

    rna_df = pd.DataFrame(
        np.random.rand(len(rna_samples), 1), index=rna_samples, columns=["feat"]
    )
    protein_df = pd.DataFrame(
        np.random.rand(len(protein_samples), 1), index=protein_samples, columns=["feat"]
    )
    meth_df = pd.DataFrame(
        np.random.rand(len(meth_samples), 1), index=meth_samples, columns=["feat"]
    )

    return DataPackage(
        multi_bulk={"rna": rna_df, "protein": protein_df, "meth": meth_df}
    )


@pytest.fixture
def config():
    return DefaultConfig()


def test_compute_membership_groups(data_package, config, sample_groups):
    splitter = PairedUnpairedSplitter(data_package, config)
    groups = splitter.membership_groups

    # Expected group keys (sorted tuples)
    expected_keys = [
        tuple(sorted(["multi_bulk.rna"])),
        tuple(sorted(["multi_bulk.protein"])),
        tuple(sorted(["multi_bulk.meth"])),
        tuple(sorted(["multi_bulk.rna", "multi_bulk.protein"])),
        tuple(sorted(["multi_bulk.rna", "multi_bulk.meth"])),
        tuple(sorted(["multi_bulk.protein", "multi_bulk.meth"])),
        tuple(sorted(["multi_bulk.rna", "multi_bulk.protein", "multi_bulk.meth"])),
    ]

    assert set(groups.keys()) == set(expected_keys)

    # Check sizes
    for key in expected_keys:
        if len(key) == 1:
            mod = key[0].split(".")[-1]
            if mod == "rna":
                assert len(groups[key]) == len(sample_groups["rna_only"])
            elif mod == "protein":
                assert len(groups[key]) == len(sample_groups["protein_only"])
            elif mod == "meth":
                assert len(groups[key]) == len(sample_groups["meth_only"])
        elif len(key) == 2:
            mods = {m.split(".")[-1] for m in key}
            if mods == {"rna", "protein"}:
                assert len(groups[key]) == len(sample_groups["rna_protein"])
            elif mods == {"rna", "meth"}:
                assert len(groups[key]) == len(sample_groups["rna_meth"])
            elif mods == {"protein", "meth"}:
                assert len(groups[key]) == len(sample_groups["protein_meth"])
        elif len(key) == 3:
            assert len(groups[key]) == len(sample_groups["rna_protein_meth"])


def test_split_consistency(data_package, config):
    splitter = PairedUnpairedSplitter(data_package, config)
    splits = splitter.split()

    modalities = ["rna", "protein", "meth"]
    split_names = ["train", "valid", "test"]

    # Helper to get sample IDs for a modality and split
    def get_ids(mod, split_name):
        indices = splits["multi_bulk"][mod][split_name]
        return set(data_package.multi_bulk[mod].index[indices])

    # Collect per-modality split assignments
    assignments = {}
    for mod in modalities:
        assignments[mod] = {}
        for split_name in split_names:
            assignments[mod][split_name] = get_ids(mod, split_name)

    # For each sample, check it's in the same split across all modalities it appears in
    all_samples = set()
    for mod in modalities:
        all_samples.update(data_package.multi_bulk[mod].index)

    for sample in all_samples:
        sample_splits = []
        for mod in modalities:
            if sample in data_package.multi_bulk[mod].index:
                for split_name in split_names:
                    if sample in assignments[mod][split_name]:
                        sample_splits.append(split_name)
                        break
        # All found splits should be the same
        assert len(set(sample_splits)) == 1, (
            f"Sample {sample} in different splits: {sample_splits}"
        )


def test_split_exclusivity(data_package, config):
    splitter = PairedUnpairedSplitter(data_package, config)
    splits = splitter.split()

    modalities = ["rna", "protein", "meth"]
    split_names = ["train", "valid", "test"]

    def get_ids(mod, split_name):
        indices = splits["multi_bulk"][mod][split_name]
        return set(data_package.multi_bulk[mod].index[indices])

    for mod in modalities:
        all_split_ids = set()
        for split_name in split_names:
            split_ids = get_ids(mod, split_name)
            # No overlap between splits
            assert all_split_ids.isdisjoint(split_ids), f"Overlap in {mod} splits"
            all_split_ids.update(split_ids)
        # All samples covered
        assert all_split_ids == set(data_package.multi_bulk[mod].index), (
            f"Not all samples assigned in {mod}"
        )


def test_split_ratios(data_package, config):
    splitter = PairedUnpairedSplitter(data_package, config)
    splits = splitter.split()

    modalities = ["rna", "protein", "meth"]

    def get_size(mod, split_name):
        return len(splits["multi_bulk"][mod][split_name])

    for mod in modalities:
        total = len(data_package.multi_bulk[mod].index)
        train_size = get_size(mod, "train")
        valid_size = get_size(mod, "valid")
        test_size = get_size(mod, "test")

        assert train_size + valid_size + test_size == total

        # Check approximate ratios (allow off-by-1 due to int flooring)
        expected_train = int(total * config.train_ratio)
        expected_valid = int(total * config.valid_ratio)
        expected_test = total - expected_train - expected_valid

        assert abs(train_size - expected_train) <= 1
        assert abs(valid_size - expected_valid) <= 1
        assert abs(test_size - expected_test) <= 1


def test_split_deterministic(data_package, config):
    # Run twice with same seed, check identical splits
    splitter1 = PairedUnpairedSplitter(data_package, config)
    splits1 = splitter1.split()

    splitter2 = PairedUnpairedSplitter(data_package, config)
    splits2 = splitter2.split()

    for parent in splits1:
        for child in splits1[parent]:
            for split_name in splits1[parent][child]:
                np.testing.assert_array_equal(
                    splits1[parent][child][split_name],
                    splits2[parent][child][split_name],
                )
