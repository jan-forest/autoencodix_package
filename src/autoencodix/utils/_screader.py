import scanpy as sc  # type: ignore
import mudata as md  # type: ignore
from anndata import AnnData  # type: ignore
from typing import Dict, Any, TYPE_CHECKING
from autoencodix.configs.default_config import DefaultConfig

if TYPE_CHECKING:
    import mudata as md  # type: ignore

    MuData = md.MuData.MuData
else:
    MuData = Any


class SingleCellDataReader:
    """Reader for multi-modal single-cell data."""

    @staticmethod
    def read_data(
        config: DefaultConfig,
    ) -> Dict[str, MuData]:  # ty: ignore[invalid-type-form]
        """Read multiple single-cell modalities into MuData object(s).

        Args:
        config: Configuration object containing data paths and parameters.

        Returns:
            For non-paired translation: Dict of Dicts with {'multi_sc': DataDict} as outer dict and with modalty keys and mudata obj as inner dict.
            For paired translation and non translation cases: dict with "multi_sc" as key and mudata as value
        """
        modalities: Dict[str, AnnData] = {}

        for mod_key, mod_info in config.data_config.data_info.items():
            if not mod_info.is_single_cell:
                continue
            adata = sc.read_h5ad(mod_info.file_path)
            modalities[mod_key] = adata

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
        return {"multi_sc": modalities}
