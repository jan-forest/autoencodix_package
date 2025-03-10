import scanpy as sc
import mudata as md
from anndata import AnnData
from typing import Dict
from autoencodix.utils.default_config import DefaultConfig


class SingleCellDataReader:
    """Reader class for multi-modal single-cell data."""

    @staticmethod
    def read_data(config: DefaultConfig) -> Dict[str, md.MuData]:
        """
        Read multiple single-cell modalities into MuData object(s).

        Parameters
        ----------
        config : DefaultConfig
            Configuration object containing data paths and parameters.

        Returns
        -------
        Dict[str, md.MuData]]
            For non-paired translation: Dict with modalty keys and mudata obj as value
            For paired translation and non translation cases: dict with "paired" as key and mudata as value
        """
        modalities: Dict[str, AnnData] = {}

        # Process each modality
        for mod_key, mod_info in config.data_config.data_info.items():
            if not mod_info.is_single_cell:
                continue
            adata = sc.read_h5ad(mod_info.file_path)
            modalities[mod_key] = adata

        if config.paired_translation is None:
            mdata = md.MuData(modalities)
            common_cells = list(
                set.intersection(
                    *(set(adata.obs_names) for adata in modalities.values())
                )
            )
            print(f"Number of common cells: {len(common_cells)}")
            mdata = mdata[common_cells]

            return {"multi_sc": mdata}
        if not config.paired_translation:
            result = {}
            for mod_key, adata in modalities.items():
                # Create individual MuData for each modality
                mdata = md.MuData({mod_key: adata})
                result[mod_key] = mdata
            print(f"Paired translation: returning {len(result)} MuData objects")
            return result
        else:
            mdata = md.MuData(modalities)
            common_cells = list(
                set.intersection(
                    *(set(adata.obs_names) for adata in modalities.values())
                )
            )
            print(f"Number of common cells: {len(common_cells)}")
            mdata = mdata[common_cells]
            return {k: v for k, v in mdata.mod.items()}
