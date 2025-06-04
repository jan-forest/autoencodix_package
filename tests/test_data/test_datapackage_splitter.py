import pytest


import numpy as np
import pandas as pd

from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.utils.default_config import (
    DefaultConfig,
    DataInfo,
    DataConfig,
    DataCase,
)
import anndata as ad  # type: ignore
import mudata as md  # type: ignore


@pytest.fixture
def unpaired_dp():
    meth_data = pd.DataFrame(
        np.random.rand(100, 50),
        index=[f"Sample_{i}" for i in range(100)],
        columns=[f"Meth_Feature_{j}" for j in range(50)],
    )

    rna_data = pd.DataFrame(
        np.random.rand(100, 100),  # TODO align
        index=[f"Sample_{i}" for i in range(100)],
        columns=[f"RNA_Feature_{j}" for j in range(100)],
    )

    clinical_annotations = pd.DataFrame(
        {
            "Sample_ID": [f"Sample_{i}" for i in range(100)],
            "Age": np.random.randint(20, 80, size=100),
            "Gender": np.random.choice(["Male", "Female"], size=100),
            "Disease_Status": np.random.choice(["Healthy", "Diseased"], size=100),
        }
    ).set_index("Sample_ID")
    dp = DataPackage()
    dp.from_modality = {"from_modality": rna_data}
    dp.to_modality = {"to_modality": meth_data}
    dp.annotation = {
        "paired": clinical_annotations
    }  # here we can also provide {RNA: rna_anno, MEth: meth_anno} if we have two anno files

    return dp


@pytest.fixture
def real_unpaired_config():
    data_info = {
        "RNA": DataInfo(translate_direction="from"),
        "METH": DataInfo(translate_direction="to"),
    }
    config = DefaultConfig(
        paired_translation=False,  # we want to have matched samples in this case
        data_case=DataCase.BULK_TO_BULK,
        data_config=DataConfig(data_info=data_info),
    )
    return config


@pytest.fixture
def dummy_dataframe():
    return pd.DataFrame({"a": range(10)})


@pytest.fixture
def dummy_datapackage(dummy_dataframe):
    return DataPackage(
        to_modality={"mod1": dummy_dataframe},
        from_modality={"mod2": dummy_dataframe},
        annotation={"to": dummy_dataframe, "from": dummy_dataframe},
        multi_sc={"rna": dummy_dataframe},
    )


@pytest.fixture
def dummy_indices():
    return {
        "train": np.array([0, 1, 2, 3, 4]),
        "valid": np.array([5, 6]),
        "test": np.array([7, 8, 9]),
    }


@pytest.fixture
def mudata_example():
    adata1 = ad.AnnData(X=np.random.rand(10, 4))
    adata1.obs["cell_type"] = ["type1"] * 5 + ["type2"] * 5
    adata1.var["gene"] = [f"gene{i}" for i in range(4)]
    adata1.obs_names = [f"cell{i}" for i in range(10)]
    adata2 = ad.AnnData(X=np.random.rand(10, 4))
    return md.MuData({"rna": adata1, "atac": adata2})


@pytest.fixture
def datapackage_with_mudata(mudata_example, dummy_dataframe):
    return DataPackage(
        to_modality={"mod1": dummy_dataframe},
        from_modality={"mod2": dummy_dataframe},
        annotation={"to": dummy_dataframe, "from": dummy_dataframe},
    )


@pytest.fixture
def paired_config():
    return DefaultConfig()


@pytest.fixture
def unpaired_config():
    return DefaultConfig(paired_translation=False)


def test_split_paired_returns_all_splits(
    dummy_datapackage, dummy_indices, paired_config
):
    splitter = DataPackageSplitter(
        dummy_datapackage, paired_config, indices=dummy_indices
    )
    result = splitter.split()
    assert set(result.keys()) == {"train", "valid", "test"}


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_split_paired_sizes(dummy_datapackage, dummy_indices, paired_config, split):
    splitter = DataPackageSplitter(
        dummy_datapackage, paired_config, indices=dummy_indices
    )
    result = splitter.split()
    assert len(result[split]["data"].to_modality["mod1"]) == len(dummy_indices[split])


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_split_paired_train_has_correct_data_type(
    dummy_datapackage, dummy_indices, paired_config, split
):
    splitter = DataPackageSplitter(
        dummy_datapackage, paired_config, indices=dummy_indices
    )
    result = splitter.split()
    assert isinstance(result[split]["data"], DataPackage)


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_split_paired_train_has_correct_indices(
    dummy_datapackage, dummy_indices, paired_config, split
):
    splitter = DataPackageSplitter(
        dummy_datapackage, paired_config, indices=dummy_indices
    )
    result = splitter.split()
    assert np.array_equal(result[split]["indices"]["paired"], dummy_indices[split])


def test_split_unpaired_missing_to_indices_raises(
    dummy_datapackage, dummy_indices, unpaired_config
):
    splitter = DataPackageSplitter(
        dummy_datapackage, unpaired_config, from_indices=dummy_indices
    )
    with pytest.raises(TypeError):
        splitter.split()


def test_split_unpaired_missing_from_indices_raises(
    dummy_datapackage, dummy_indices, unpaired_config
):
    splitter = DataPackageSplitter(
        dummy_datapackage, unpaired_config, to_indices=dummy_indices
    )
    with pytest.raises(TypeError):
        splitter.split()


def test_split_unpaired_returns_split(unpaired_dp, dummy_indices, real_unpaired_config):
    splitter = DataPackageSplitter(
        unpaired_dp,
        real_unpaired_config,
        to_indices=dummy_indices,
        from_indices=dummy_indices,
    )
    result = splitter.split()
    print(result)
    assert set(result.keys()) == {"train", "valid", "test"}


def test_split_unpaired_train_has_correct_modalities(
    unpaired_dp, dummy_indices, real_unpaired_config
):
    splitter = DataPackageSplitter(
        unpaired_dp,
        real_unpaired_config,
        to_indices=dummy_indices,
        from_indices=dummy_indices,
    )
    result = splitter.split()
    assert isinstance(result["train"]["data"], DataPackage)


def test_split_with_none_datapackage_raises(paired_config):
    with pytest.raises(TypeError):
        splitter = DataPackageSplitter(None, paired_config, indices={})
        splitter.split()
