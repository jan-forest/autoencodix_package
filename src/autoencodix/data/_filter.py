"""
Ignoring types for scipy and sklearn, because it stubs require Python > 3.9, which we want to allow
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Callable
from scipy.stats import median_abs_deviation  # type: ignore
from sklearn.preprocessing import (  # type: ignore
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)  # type: ignore
from enum import Enum
from autoencodix.utils.default_config import DataInfo
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore


class FilterMethod(Enum):
    """Supported filtering methods.

    Attributes:
        VARCORR (str): Filter by variance and correlation.
        NOFILT (str): No filtering.
        VAR (str): Filter by variance.
        MAD (str): Filter by median absolute deviation.
        NONZEROVAR (str): Filter by non-zero variance.
        CORR (str): Filter by correlation.
    """

    VARCORR = "VARCORR"
    NOFILT = "NOFILT"
    VAR = "VAR"
    MAD = "MAD"
    NONZEROVAR = "NONZEROVAR"
    CORR = "CORR"


class DataFilter:
    """A class for filtering features from dataframes.

    This class provides methods to filter features in a DataFrame using various
    statistical approaches such as variance, correlation, and median absolute deviation.

    Attributes:
        df (pd.DataFrame): Input dataframe to be filtered.
        data_info (DataInfo): Configuration object containing filtering parameters.
        _filter_pipelines (Dict[FilterMethod, List[Callable]]): Dictionary mapping filter methods to sequences of filter functions.
    """

    def __init__(self, df: pd.DataFrame, data_info: DataInfo):
        """Initialize the DataFilter with a dataframe and configuration.

        Parameters:
            df (pd.DataFrame): Input dataframe to be filtered.
            data_info (DataInfo): Configuration object containing filtering parameters.
        """
        self.df = df
        self.data_info = data_info
        self._initialize_filter_pipeline()

    def _initialize_filter_pipeline(self) -> None:
        """Initialize the filter pipeline based on the available filter methods.

        Sets up the _filter_pipelines dictionary that maps each FilterMethod to a
        sequence of filter functions that will be applied in order.
        """
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
        """Remove features with zero variance.

        Parameters:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Filtered dataframe containing only columns with non-zero variance.
        """
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var[var > 0].index]

    @staticmethod
    def _filter_by_variance(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
        """Keep top k features by variance.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            k (Optional[int]): Number of top variance features to keep. If None or greater
                              than number of columns, all features are kept.

        Returns:
            pd.DataFrame: Filtered dataframe with top k variance features.
        """
        if k is None or k > df.shape[1]:
            return df
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var.sort_values(ascending=False).index[:k]]

    @staticmethod
    def _filter_by_mad(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
        """Keep top k features by median absolute deviation.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            k (Optional[int]): Number of top MAD features to keep. If None or greater
                              than number of columns, all features are kept.

        Returns:
            pd.DataFrame: Filtered dataframe with top k MAD features.
        """
        if k is None or k > df.shape[1]:
            return df
        mads = pd.Series(median_abs_deviation(df, axis=0), index=df.columns)
        return df[mads.sort_values(ascending=False).index[:k]]

    def _filter_by_correlation(
        self, df: pd.DataFrame, k: Optional[int]
    ) -> pd.DataFrame:
        """Filter features using correlation-based clustering.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            k (Optional[int]): Number of clusters to create. If None or greater
                              than number of columns, all features are kept.

        Returns:
            pd.DataFrame: Filtered dataframe with one representative feature per cluster.

        Raises:
            NotImplementedError: This method is not implemented in the instance method version.
        """
        if k is None or k > df.shape[1]:
            return df
        else:
            raise NotImplementedError("Correlation-based filtering not implemented")

    @staticmethod
    def filter_by_correlation(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
        """Filter features using correlation-based clustering.

        This method clusters features based on their correlation distance and
        selects a representative feature (medoid) from each cluster.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            k (Optional[int]): Number of clusters to create. If None or greater
                              than number of columns, all features are kept.

        Returns:
            pd.DataFrame: Filtered dataframe with one representative feature (medoid) per cluster.
        """
        if k is None or k > df.shape[1]:
            return df
        else:
            X = df.transpose().values

            dist_matrix = squareform(pdist(X, metric="correlation"))

            clustering = AgglomerativeClustering(
                n_clusters=k,
                affinity="precomputed",
                linkage="average",  # Average linkage is often used with correlation
            ).fit(dist_matrix)

            # Find the medoid of each cluster
            medoid_indices = []
            for i in range(k):
                cluster_points = np.where(clustering.labels_ == i)[0]
                if len(cluster_points) > 0:
                    # The medoid is the point with minimum sum of distances to other points in the cluster
                    cluster_dist_matrix = dist_matrix[
                        np.ix_(cluster_points, cluster_points)
                    ]
                    sum_distances = np.sum(cluster_dist_matrix, axis=1)
                    medoid_idx = cluster_points[np.argmin(sum_distances)]
                    medoid_indices.append(medoid_idx)

            # Filter the DataFrame using the medoid indices
            df_filt = df.iloc[:, medoid_indices]
            return df_filt

    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale data using the specified method in data_info.

        Parameters:
            df (pd.DataFrame): Input dataframe to be scaled.

        Returns:
            pd.DataFrame: Scaled dataframe.
        """
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
        """Apply the configured filtering method to the dataframe.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        MIN_FILTER = 10
        print(f"Applying {self.data_info.filtering} filtering")
        if self.df.shape[0] < MIN_FILTER or self.df.empty:
            print(
                f"WARNING: df is too small for filtering, needs to have at least {MIN_FILTER}"
            )
            return self.df

        filtered_df = self.df.copy()
        mapped_filtering = FilterMethod(self.data_info.filtering)

        for filter_step in self._filter_pipelines[mapped_filtering]:
            prev_shape = filtered_df.shape
            filtered_df = filter_step(filtered_df)
            print(f"Shape changed from {prev_shape} to {filtered_df.shape}")

        return filtered_df

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured scaling method to the dataframe.

        Parameters:
            df (pd.DataFrame): Input dataframe to be scaled.

        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        print(f"Applying {self.data_info.scaling} scaling")
        return self._scale_data(df=df)

    @property
    def available_methods(self) -> List[str]:
        """List all available filtering methods.

        Returns:
            List[str]: List of available filtering method names.
        """
        return [method.value for method in FilterMethod]
