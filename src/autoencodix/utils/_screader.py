import scanpy as sc
import mudata as md
from anndata import AnnData
from typing import Dict
from autoencodix.utils.default_config import DefaultConfig


class SingleCellDataReader:
    """Reader class for multi-modal single-cell data."""

    @staticmethod
    def read_data(config: DefaultConfig) -> md.MuData:
        """
        Read multiple single-cell modalities into a MuData object.

        Parameters
        ----------
        config : DefaultConfig
            Configuration object containing data paths and parameters.

        Returns
        -------
        MuData
            Multi-modal data object containing all modalities.
        """
        modalities: Dict[str, AnnData] = {}

        # Process each modality
        for mod_key, mod_info in config.data_config.data_info.items():
            if not mod_info.is_single_cell:
                continue
            adata = sc.read_h5ad(mod_info.file_path)
            modalities[mod_key] = adata

        mdata = md.MuData(modalities)

        if not config.paired_translation:
            common_cells = list(
                set.intersection(
                    *(set(adata.obs_names) for adata in modalities.values())
                )
            )
            print(f"Number of common cells: {len(common_cells)}")
            mdata = mdata[common_cells]

        return mdata
