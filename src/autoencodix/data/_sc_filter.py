import mudata as md
import scanpy as sc
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Union
from autoencodix.data._filter import DataFilter


class SingleCellFilter:
    """Filter and scale single-cell data, returning a MuData object with synchronized metadata."""

    def __init__(
        self, mudata: md.MuData, data_info: Union[BaseModel, Dict[str, BaseModel]]
    ):
        """
        Initialize single-cell filter.
        Parameters
        ----------
        mudata : MuData
            Multi-modal data to be filtered
        data_info : Union[BaseModel, Dict[str, BaseModel]]
            Either a single data_info object for all modalities or a dictionary of data_info objects
            for each modality.
        """
        self.mudata = mudata
        self.data_info = data_info
        self._is_data_info_dict = isinstance(data_info, dict)

    def _get_data_info_for_modality(self, mod_key: str) -> BaseModel:
        """
        Get the data_info configuration for a specific modality.
        Parameters
        ----------
        mod_key : str
            The modality key (e.g., "RNA", "METH")
        Returns
        -------
        BaseModel
            The data_info configuration for the modality
        """
        if self._is_data_info_dict:
            return self.data_info.get(mod_key)
        else:
            return self.data_info

    def _preprocess(self) -> md.MuData:
        """
        Preprocess the data using modality-specific configurations.
        Returns
        -------
        MuData
            Preprocessed data
        """
        mudata = self.mudata.copy()
        for mod_key, mod_data in mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is not None:
                sc.pp.filter_cells(mod_data, min_genes=data_info.min_genes)
                sc.pp.filter_genes(mod_data, min_cells=data_info.min_cells)
                if data_info.normalize_counts:
                    sc.pp.normalize_total(mod_data)
                if data_info.log_transform:
                    sc.pp.log1p(mod_data)
                mudata.mod[mod_key] = mod_data
        return mudata

    def _to_dataframe(self, mod_data) -> pd.DataFrame:
        """
        Transform a modality's AnnData object to a pandas DataFrame.
        Parameters
        ----------
        mod_data : AnnData
            Modality data to be transformed
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        return pd.DataFrame(
            mod_data.X.toarray(), columns=mod_data.var_names, index=mod_data.obs_names
        )

    def _from_dataframe(self, df: pd.DataFrame, mod_data):
        """
        Update a modality's AnnData object with the values from a DataFrame.
        This also synchronizes the `obs` and `var` metadata to match the filtered data.
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the updated values
        mod_data : AnnData
            Modality data to be updated
        """
        # Filter the AnnData object to match the rows and columns of the DataFrame
        mod_data = mod_data[df.index, df.columns].copy()
        # Update the data matrix with the filtered and scaled values
        mod_data.X = df.values
        return mod_data

    def apply_general_filtering_and_scaling(self, mod_data, data_info):
        """
        Apply general filtering and scaling to a modality's data.
        Parameters
        ----------
        mod_data : AnnData
            Modality data to be filtered and scaled
        data_info : BaseModel
            Configuration for filtering and scaling
        Returns
        -------
        AnnData
            Filtered and scaled modality data with synchronized metadata
        """
        df = self._to_dataframe(mod_data)
        data_filter = DataFilter(df, data_info)
        filtered_df = data_filter.filter()
        scaled_df = data_filter.scale(filtered_df)
        # Update the AnnData object with the filtered and scaled data
        mod_data = self._from_dataframe(scaled_df, mod_data)
        return mod_data

    def preprocess(self) -> md.MuData:
        """
        Process the single-cell data by preprocessing, filtering, and scaling.
        Returns
        -------
        MuData
            Processed MuData object with filtered and scaled values, and synchronized metadata
        """
        preprocessed_mudata = self._preprocess()
        for mod_key, mod_data in preprocessed_mudata.mod.items():
            data_info = self._get_data_info_for_modality(mod_key)
            if data_info is not None:
                updated_mod_data = self.apply_general_filtering_and_scaling(
                    mod_data, data_info
                )
                preprocessed_mudata.mod[mod_key] = updated_mod_data
        return preprocessed_mudata
