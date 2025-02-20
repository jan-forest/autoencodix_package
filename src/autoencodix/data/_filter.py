import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Callable
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from enum import Enum
from autoencodix.utils.default_config import DataInfo


class FilterMethod(Enum):
    """Supported filtering methods"""
    VARCORR = "VARCORR"
    NOFILT = "NOFILT"
    VAR = "VAR"
    MAD = "MAD"
    NONZEROVAR = "NONZEROVAR"
    CORR = "CORR"


class DataFilter:
    """A class for filtering features."""

    def __init__(self, df: pd.DataFrame, data_info: DataInfo):
        self.df = df
        self.data_info = data_info
        self._initialize_filter_pipeline()

    def _initialize_filter_pipeline(self) -> None:
        self._filter_pipelines: Dict[FilterMethod, List[Callable]] = {
            FilterMethod.VARCORR: [
                self._filter_nonzero_variance,
                lambda df: self._filter_by_variance(
                    df,
                    self.data_info.k_filter * 10 if self.data_info.k_filter else None,
                ),
                lambda df: self._filter_by_correlation(
                    df, self.data_info.k_filter if self.data_info.k_filter else None
                ),
            ],
            FilterMethod.VAR: [
                self._filter_nonzero_variance,
                lambda df: self._filter_by_variance(
                    df, self.data_info.k_filter if self.data_info.k_filter else None
                ),
            ],
            FilterMethod.MAD: [
                self._filter_nonzero_variance,
                lambda df: self._filter_by_mad(
                    df, self.data_info.k_filter if self.data_info.k_filter else None
                ),
            ],
            FilterMethod.CORR: [
                self._filter_nonzero_variance,
                lambda df: self._filter_by_correlation(
                    df, self.data_info.k_filter if self.data_info.k_filter else None
                ),
            ],
            FilterMethod.NONZEROVAR: [self._filter_nonzero_variance],
            FilterMethod.NOFILT: [],
        }

    @staticmethod
    def _filter_nonzero_variance(df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance"""
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var[var > 0].index]

    @staticmethod
    def _filter_by_variance(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
        """Keep top k features by variance"""
        if k is None or k > df.shape[1]:
            return df
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var.sort_values(ascending=False).index[:k]]

    @staticmethod
    def _filter_by_mad(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
        """Keep top k features by median absolute deviation"""
        if k is None or k > df.shape[1]:
            return df
        mads = pd.Series(median_abs_deviation(df, axis=0), index=df.columns)
        return df[mads.sort_values(ascending=False).index[:k]]

    def _filter_by_correlation(
        self, df: pd.DataFrame, k: Optional[int]
    ) -> pd.DataFrame:
        """Filter features using correlation-based clustering."""
        if k is None or k > df.shape[1]:
            return df
        else:
            raise NotImplementedError("Correlation-based filtering not implemented")

    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale data using the specified method in data_info."""
        method = self.data_info.scaling.upper()

        if method == "MINMAX":
            scaler = MinMaxScaler(clip=True)
        elif method == "STANDARD":
            scaler = StandardScaler()
        elif method == "ROBUST":
            scaler = RobustScaler()
        elif method == "MAXABS":
            scaler = MaxAbsScaler()
        else:
            return df  # No scaling applied if method is "NONE" or unrecognized

        df_scaled = pd.DataFrame(
            scaler.fit_transform(df), columns=df.columns, index=df.index
        )
        return df_scaled

    def filter(self) -> pd.DataFrame:
        """Apply the configured filtering method."""

        print(f"Applying {self.data_info.filtering} filtering")

        filtered_df = self.df.copy()
        mapped_filtering = FilterMethod(self.data_info.filtering)

        for filter_step in self._filter_pipelines[mapped_filtering]:
            prev_shape = filtered_df.shape
            filtered_df = filter_step(filtered_df)
            print(f"Shape changed from {prev_shape} to {filtered_df.shape}")


        return filtered_df
    
    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured scaling method."""

        print(f"Applying {self.data_info.scaling} scaling")
        return self._scale_data(df=df)

    @property
    def available_methods(self) -> List[str]:
        """List all available filtering methods"""
        return [method.value for method in FilterMethod]
