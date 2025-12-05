# test_data_filter.py

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)
from autoencodix.configs.default_config import DataInfo, DefaultConfig, DataConfig
from autoencodix.data._filter import DataFilter, FilterMethod


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(20, 10), columns=[f"gene_{i}" for i in range(10)]
    )


@pytest.fixture
def df_with_zero_variance():
    df = pd.DataFrame(np.random.rand(20, 9), columns=[f"gene_{i}" for i in range(9)])
    df["constant"] = 1.0
    return df


@pytest.mark.parametrize(
    "method,k_filter,expected_cols",
    [
        ("NOFILT", 5, 10),
        ("VAR", 5, 5),
        ("MAD", 5, 5),
        ("CORR", 5, 5),
        ("VARCORR", 3, 3),
    ],
)
def test_filter_methods(
    sample_df, df_with_zero_variance, method, k_filter, expected_cols
):
    df = df_with_zero_variance if method == "NONZEROVAR" else sample_df
    data_info = DataInfo(filtering=method, scaling="NONE")
    config = DefaultConfig(
        k_filter=k_filter, data_config=DataConfig(data_info={"df": data_info})
    )
    config.data_config.data_info["df"].k_filter = k_filter
    dfilt = DataFilter(data_info, config=config)
    filtered_df, _ = dfilt.filter(df)
    assert filtered_df.shape[1] == expected_cols


def test_filter_with_genes_to_keep(sample_df):
    data_info = DataInfo(filtering="VAR", scaling="NONE")
    config = DefaultConfig(k_filter=5)
    dfilt = DataFilter(data_info, config=config)

    genes = sample_df.columns[:3].tolist()
    filtered_df, kept = dfilt.filter(sample_df, genes_to_keep=genes)
    assert kept == genes


def test_filter_with_missing_genes_raises_keyerror(sample_df):
    data_info = DataInfo(filtering="VAR", scaling="NONE")
    config = DefaultConfig(k_filter=5)
    dfilt = DataFilter(data_info, config=config)

    with pytest.raises(KeyError):
        dfilt.filter(sample_df, genes_to_keep=["not_a_gene"])


@pytest.mark.parametrize(
    "scaling_method,expected_type",
    [
        ("MINMAX", MinMaxScaler),
        ("STANDARD", StandardScaler),
        ("ROBUST", RobustScaler),
        ("MAXABS", MaxAbsScaler),
        ("NONE", None),
    ],
)
def test_scaler_initialization(sample_df, scaling_method, expected_type):
    data_info = DataInfo(filtering="NOFILT", scaling=scaling_method)
    config = DefaultConfig(k_filter=None)
    dfilt = DataFilter(data_info, config=config)

    scaler = dfilt.fit_scaler(sample_df)
    print(f"Scaler type: {type(scaler)}")
    if expected_type is None:
        assert scaler is None
    else:
        assert isinstance(scaler, expected_type)


def test_scale_applies_scaler(sample_df):
    data_info = DataInfo(filtering="NOFILT", scaling="MINMAX")
    config = DefaultConfig(k_filter=None)
    dfilt = DataFilter(data_info, config=config)

    scaler = dfilt.fit_scaler(sample_df)
    scaled = dfilt.scale(sample_df, scaler)
    assert np.allclose(scaled.min().min(), 0.0, atol=1e-6)


def test_available_methods_matches_enum():
    methods = DataFilter(
        DataInfo(filtering="NOFILT", scaling="NONE"),
        config=DefaultConfig(k_filter=None),
    ).available_methods
    enum_values = [e.value for e in FilterMethod]
    assert set(methods) == set(enum_values)
