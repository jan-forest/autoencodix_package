"""
Ignoring types for scipy and sklearn, because it stubs require Python > 3.9, which we want to allow
"""

import pandas as pd
import warnings
import numpy as np
from typing import Optional, List, Set, Tuple, Any, Union
from scipy.stats import median_abs_deviation  # type: ignore
from sklearn.preprocessing import (  # type: ignore
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)  # type: ignore
from enum import Enum
from autoencodix.configs.default_config import DataInfo, DefaultConfig
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore


class FilterMethod(Enum):
    """Supported filtering methods.

    Attributes:
        VARCORR: Filter by variance and correlation.
        NOFILT No filtering.
        VAR: Filter by variance.
        MAD: Filter by median absolute deviation.
        NONZEROVAR: Filter by non-zero variance.
        CORR: Filter by correlation.
    """

    VARCORR = "VARCORR"
    NOFILT = "NOFILT"
    VAR = "VAR"
    MAD = "MAD"
    NONZEROVAR = "NONZEROVAR"
    CORR = "CORR"


class DataFilter:
    """Preprocesses dataframes, including filtering and scaling.

    This class separates the filtering logic that needs to be applied consistently
    across train, validation, and test sets from the scaling logic that is
    typically fitted on the training data and then applied to the other sets.

    Attributes:
        data_info: Configuration object containing preprocessing parameters.
        filtered_features: Set of features to keep after filtering on the training data. None initially.
        _scaler: The fitted scaler object. None initially.
        ontologies: Ontology information, if provided for Ontix.
        config: Configuration object containing default parameters.
    """

    def __init__(
        self,
        data_info: DataInfo,
        config: DefaultConfig,
        ontologies: Optional[tuple] = None,
    ):  # Addition to Varix, mandotory for Ontix
        """Initializes the DataFilter with a configuration.

        Args:
            data_info: Configuration object containing preprocessing parameters.
            config: Configuration object containing default parameters.
            ontologies: Ontology information, if provided for Ontix.
        """
        self.data_info = data_info
        self.config = config
        self.filtered_features: Optional[Set[str]] = None
        self._scaler = None
        self.ontologies = ontologies  # Addition to Varix, mandotory for Ontix

    def _filter_nonzero_variance(self, df: pd.DataFrame) -> pd.Series:
        """Removes features with zero variance.

        Args:
            df: Input dataframe.

        Returns:
            Filtered dataframe containing only columns with non-zero variance.
        """
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var[var > 0].index]

    def _filter_by_variance(
        self, df: pd.DataFrame, k: Optional[int]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Keeps top k features by variance.

        Args:
            df: Input dataframe.
            k: Number of top variance features to keep. If None or greater
               than number of columns, all features are kept.

        Returns:
            Filtered dataframe with top k variance features.
        """
        if k is None or k > df.shape[1]:
            warnings.warn(
                "WARNING: k is None or greater than number of columns, keeping all features."
            )
            return df
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var.sort_values(ascending=False).index[:k]]

    def _filter_by_mad(
        self, df: pd.DataFrame, k: Optional[int]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Keeps top k features by median absolute deviation.

        Args:
            df: Input dataframe.
            k: Number of top MAD features to keep. If None or greater
               than number of columns, all features are kept.

        Returns:
            Filtered dataframe with top k MAD features.
        """
        if k is None or k > df.shape[1]:
            return df
        mads = pd.Series(median_abs_deviation(df, axis=0), index=df.columns)
        return df[mads.sort_values(ascending=False).index[:k]]

    def _filter_by_correlation(
        self, df: pd.DataFrame, k: Optional[int]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Filters features using correlation-based clustering.

        This method clusters features based on their correlation distance and
        selects a representative feature (medoid) from each cluster.

        Args:
            df: Input dataframe.
            k: Number of clusters to create. If None or greater
               than number of columns, all features are kept.

        Returns:
            Filtered dataframe with one representative feature (medoid) per cluster.
        """
        if k is None or k > df.shape[1]:
            warnings.warn(
                "WARNING: k is None or greater than number of columns, keeping all features."
            )
            return df
        else:
            X = df.transpose().values

            dist_matrix = squareform(pdist(X, metric="correlation"))

            clustering = AgglomerativeClustering(
                n_clusters=k,
            ).fit(dist_matrix)

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

            df_filt: Union[pd.DataFrame, pd.Series] = df.iloc[:, medoid_indices]
            return df_filt

    def filter(
        self, df: pd.DataFrame, genes_to_keep: Optional[List] = None
    ) -> Tuple[Union[pd.Series, pd.DataFrame], List[str]]:
        """Applies the configured filtering method to the dataframe.

        This method is intended to be called on the training data to determine
        which features to keep. The `filtered_features` attribute will be set
        based on the result.

        Args:
            df: Input dataframe to be filtered (typically the training set).
            genes_to_keep: A list of gene names to explicitly keep.
                If provided, other filtering methods will be ignored.

        Returns:
            A tuple containing:
                - The filtered dataframe.
                - A list of column names (features) that were kept.

        Raises:
            KeyError: If some genes in `genes_to_keep` are not present in the dataframe.
        """
        if genes_to_keep is not None:
            try:
                df: Union[pd.Series, pd.DataFrame] = df[genes_to_keep]
                return df, genes_to_keep
            except KeyError as e:
                raise KeyError(
                    f"Some genes in genes_to_keep are not present in the dataframe: {e}"
                )

        MIN_FILTER = 2
        filtering_method = FilterMethod(self.data_info.filtering)

        if df.shape[0] < MIN_FILTER or df.empty:
            warnings.warn(
                f"WARNING: df is too small for filtering, needs to have at least {MIN_FILTER}"
            )
            return df, df.columns.tolist()

        filtered_df = df.copy()

        ## Remove features which are not in the ontology for Ontix architecture
        ## must be done before other filtering is applied
        if hasattr(self, "ontologies") and self.ontologies is not None:
            all_feature_names: Union[Set, List] = set()
            for key, values in self.ontologies[-1].items():
                all_feature_names.update(values)
            all_feature_names = list(all_feature_names)
            feature_order = filtered_df.columns.tolist()
            missing_features = [f for f in feature_order if f not in all_feature_names]
            ## Filter out features not in the ontology
            feature_order = [f for f in feature_order if f in all_feature_names]
            if missing_features:
                print(
                    f"Features in feature_order not found in all_feature_names: {missing_features}"
                )

            filtered_df = filtered_df.loc[:, feature_order]

        ####

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

        return filtered_df, filtered_df.columns.tolist()

    def _init_scaler(self) -> None:
        """Initializes the scaler based on the configured scaling method."""
        method = self.data_info.scaling

        if method == "NOTSET":
            # if not set in data config, we use the global scaling config
            method = self.config.scaling
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

    def fit_scaler(self, df: Union[pd.Series, pd.DataFrame]) -> Any:
        """Fits the scaler to the input dataframe (typically the training set).

        Args:
            df: Input dataframe to fit the scaler on.

        Returns:
            The fitted scaler object.
        """
        self._init_scaler()
        if self._scaler is not None:
            self._scaler.fit(df)
        else:
            warnings.warn("No scaling applied.")
        return self._scaler

    def scale(
        self, df: Union[pd.Series, pd.DataFrame], scaler: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """Applies the fitted scaler to the input dataframe.

        Args:
            df: Input dataframe to be scaled.
            scaler: The fitted scaler object.

        Returns:
            Scaled dataframe.
        """
        if scaler is None:
            warnings.warn("No scaler has been fitted yet or scaling is set to none.")
            return df

        df_scaled = pd.DataFrame(
            scaler.transform(df), columns=df.columns, index=df.index
        )
        return df_scaled

    @property
    def available_methods(self) -> List[str]:
        """Lists all available filtering methods.

        Returns:
            List of available filtering method names.
        """
        return [method.value for method in FilterMethod]
