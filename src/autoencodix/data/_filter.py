"""
Ignoring types for scipy and sklearn, because it stubs require Python > 3.9, which we want to allow
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Set, Tuple, Any
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
    """A class for preprocessing dataframes, including filtering and scaling.

    This class separates the filtering logic that needs to be applied consistently
    across train, validation, and test sets from the scaling logic that is
    typically fitted on the training data and then applied to the other sets.

    Attributes:
        data_info (DataInfo): Configuration object containing preprocessing parameters.
        filtered_features (Optional[Set[str]]): Set of features to keep after filtering
                                                 on the training data. None initially.
    """

    def __init__(self, data_info: DataInfo):
        """Initialize the DataPreprocessor with a configuration.

        Parameters:
            data_info (DataInfo): Configuration object containing preprocessing parameters.
        """
        self.data_info = data_info
        self.filtered_features: Optional[Set[str]] = None
        self._scaler = None

    def _filter_nonzero_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance.

        Parameters:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Filtered dataframe containing only columns with non-zero variance.
        """
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var[var > 0].index]

    def _filter_by_variance(self, df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
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

    def _filter_by_mad(self, df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
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

    def filter(
        self, df: pd.DataFrame, genes_to_keep: Optional[List] = None
    ) -> Tuple[
        pd.DataFrame, List[str]
    ]:  # TODO add indicies and update for the ones that got dropped
        """Apply the configured filtering method to the dataframe.

        This method is intended to be called on the training data to determine
        which features to keep. The `filtered_features` attribute will be set
        based on the result.

        Parameters:
            df (pd.DataFrame): Input dataframe to be filtered (typically the training set).

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        if genes_to_keep is not None:
            try:
                df = df[genes_to_keep]
                return df, genes_to_keep
            except KeyError as e:
                raise KeyError(
                    f"Some genes in genes_to_keep are not present in the dataframe: {e}"
                )

        MIN_FILTER = 10
        filtering_method = FilterMethod(self.data_info.filtering)
        print(f"Applying {filtering_method.value} filtering")

        if df.shape[0] < MIN_FILTER or df.empty:
            print(
                f"WARNING: df is too small for filtering, needs to have at least {MIN_FILTER}"
            )
            return df, df.columns.tolist()

        filtered_df = df.copy()

        if filtering_method == FilterMethod.NOFILT:
            return filtered_df, df.columns.tolist()
        if self.data_info.k_filter is None:
            return filtered_df, df.columns.tolist()

        if filtering_method == FilterMethod.NONZEROVAR:
            filtered_df = self._filter_nonzero_variance(filtered_df)
        elif filtering_method == FilterMethod.VAR:
            filtered_df = self._filter_nonzero_variance(filtered_df)
            filtered_df = self._filter_by_variance(filtered_df, self.data_info.k_filter)
        elif filtering_method == FilterMethod.MAD:
            filtered_df = self._filter_nonzero_variance(filtered_df)
            filtered_df = self._filter_by_mad(filtered_df, self.data_info.k_filter)
        elif filtering_method == FilterMethod.CORR:
            filtered_df = self._filter_nonzero_variance(filtered_df)
            filtered_df = self._filter_by_correlation(
                filtered_df, self.data_info.k_filter
            )
        elif filtering_method == FilterMethod.VARCORR:
            filtered_df = self._filter_nonzero_variance(filtered_df)
            filtered_df = self._filter_by_variance(
                filtered_df,
                self.data_info.k_filter * 10 if self.data_info.k_filter else None,
            )
            if self.data_info.k_filter is not None:
                # Apply correlation filter on the already variance-filtered data
                num_features_after_var = filtered_df.shape[1]
                k_corr = min(self.data_info.k_filter, num_features_after_var)
                filtered_df = self._filter_by_correlation(filtered_df, k_corr)

        print(f"Shape after filtering: {filtered_df.shape}")
        return filtered_df, filtered_df.columns.tolist()

    def _init_scaler(self) -> None:
        """Initialize the scaler based on the configured scaling method."""
        method = self.data_info.scaling
        if method == "MINMAX":
            self._scaler = MinMaxScaler(clip=True)
        elif method == "STANDARD":
            self._scaler = StandardScaler()
        elif method == "ROBUST":
            self._scaler = RobustScaler()
        elif method == "MAXABS":
            self._scaler = MaxAbsScaler()
        else:
            self._scaler = None

    def fit_scaler(self, df: pd.DataFrame) -> Any:
        """Fit the scaler to the input dataframe (typically the training set).

        Parameters:
            df (pd.DataFrame): Input dataframe to fit the scaler on.
        """
        self._init_scaler()
        if self._scaler is not None:
            self._scaler.fit(df)
            print(f"Fitted {self.data_info.scaling} scaler.")
        else:
            print("No scaling applied.")
        return self._scaler

    def scale(self, df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        """Apply the fitted scaler to the input dataframe.

        Parameters:
            df (pd.DataFrame): Input dataframe to be scaled.

        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        if scaler is None:
            print("No scaler has been fitted yet or scaling is set to NONE.")
            return df

        print(f"Applying {self.data_info.scaling} scaling.")
        print(f"shape before scaling: {df.shape}")
        print(f"scaler: {scaler}")
        df_scaled = pd.DataFrame(
            scaler.transform(df), columns=df.columns, index=df.index
        )
        return df_scaled

    @property
    def available_methods(self) -> List[str]:
        """List all available filtering methods.

        Returns:
            List[str]: List of available filtering method names.
        """
        return [method.value for method in FilterMethod]
