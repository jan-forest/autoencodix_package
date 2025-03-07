import scanpy as sc
import numpy as np
from mudata import MuData
from anndata import AnnData
from scipy.sparse import issparse


class SingleCellFilter:
    """Filter single-cell data."""

    def __init__(self, mudata, data_info):
        """
        Initialize single-cell filter.

        Parameters
        ----------
        mudata : MuData
            Multi-modal data to be filtered
        data_info : DataInfo
            Information about the data modality
        """
        self.mudata = mudata
        self.data_info = data_info

    def _ensure_no_nans(self, adata):
        """Ensure there are no NaN values in the AnnData object."""
        X = adata.X
        if issparse(X):
            X_dense = X.data
        else:
            X_dense = X

        if np.isnan(X_dense).any():
            print("Warning: Found NaN values in data matrix. Replacing with zeros.")
            if issparse(X):
                X.data[np.isnan(X.data)] = 0
            else:
                X[np.isnan(X)] = 0
            adata.X = X

        for layer_name, layer_data in adata.layers.items():
            if issparse(layer_data):
                layer_dense = layer_data.data
            else:
                layer_dense = layer_data

            if np.isnan(layer_dense).any():
                print(
                    f"Warning: Found NaN values in layer {layer_name}. Replacing with zeros."
                )
                if issparse(layer_data):
                    layer_data.data[np.isnan(layer_data.data)] = 0
                else:
                    layer_data[np.isnan(layer_data)] = 0
                adata.layers[layer_name] = layer_data

        return adata

    def preprocess(self):
        """
        Preprocess the data.

        Returns
        -------
        MuData
            Preprocessed data
        """
        mudata = self.mudata.copy()

        for mod_key, mod_data in mudata.mod.items():
            print(f"Processing modality {mod_key}")
            mod_data = self._ensure_no_nans(mod_data)

            if self.data_info.min_genes > 0:
                print(
                    f"Filtering cells with fewer than {self.data_info.min_genes} genes in modality {mod_key}"
                )
                sc.pp.filter_cells(mod_data, min_genes=self.data_info.min_genes)

            if self.data_info.min_cells > 0:
                print(
                    f"Filtering genes detected in fewer than {self.data_info.min_cells} cells in modality {mod_key}"
                )
                sc.pp.filter_genes(mod_data, min_cells=self.data_info.min_cells)

            if self.data_info.normalize_counts:
                print(f"Normalizing in modality {mod_key}")
                sc.pp.normalize_total(mod_data)

            if self.data_info.log_transform:
                print(f"Applying log transformation in modality {mod_key}")
                sc.pp.log1p(mod_data)

            if self.data_info.selected_layers:
                for layer in self.data_info.selected_layers:
                    if layer in mod_data.layers:
                        print(f"Processing layer {layer} in modality {mod_key}")
                        # Implement layer-specific processing here

            mudata.mod[mod_key] = mod_data

        return mudata
