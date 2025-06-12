import scanpy as sc  # type: ignore
import mudata as md  # type: ignore
from anndata import AnnData  # type: ignore
from typing import Dict, Any, TYPE_CHECKING, Optional, List
from autoencodix.utils.default_config import DefaultConfig

if TYPE_CHECKING:
    import mudata as md  # type: ignore

    MuData = md.MuData.MuData
else:
    MuData = Any


class SingleCellDataReader:
    """Reader class for multi-modal single-cell data."""

    @staticmethod
    def read_data(config: DefaultConfig) -> Dict[str, MuData]:
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
        is_translation: bool = False

        # Process each modality
        for mod_key, mod_info in config.data_config.data_info.items():
            if mod_info.translate_direction is not None:
                is_translation = True
            if not mod_info.is_single_cell:
                continue
            adata = sc.read_h5ad(mod_info.file_path)
            modalities[mod_key] = adata

        if is_translation:
            return {"multi_sc": modalities}
        if config.requires_paired:
            mdata = md.MuData(modalities)
            common_cells = list(
                set.intersection(
                    *(set(adata.obs_names) for adata in modalities.values())
                )
            )
            print(f"Number of common cells: {len(common_cells)}")
            mdata = mdata[common_cells]
            return {"multi_sc": mdata}
        # unpaired non-translation case
        else:
            result = {}
            for mod_key, adata in modalities.items():
                # Create individual MuData for each modality
                mdata = md.MuData({mod_key: adata})
                result[mod_key] = mdata
            print(f"Unpaired None translation: returning {len(result)} MuData objects")
            return result
