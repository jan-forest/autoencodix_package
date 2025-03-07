import scanpy as sc
import mudata as md
from anndata import AnnData
from typing import Dict, Optional, Union
from autoencodix.utils.default_config import DefaultConfig

class SingleCellDataReader:
    """Reader class for multi-modal single-cell data."""
    
    @staticmethod
    def read_data(config: DefaultConfig) -> Union[md.MuData, Dict[str, md.MuData]]:
        """
        Read multiple single-cell modalities into MuData object(s).
        
        Parameters
        ----------
        config : DefaultConfig
            Configuration object containing data paths and parameters.
            
        Returns
        -------
        Union[md.MuData, Dict[str, md.MuData]]
            For non-paired translation: Multi-modal data object containing all modalities.
            For paired translation: Dictionary mapping modality keys to individual MuData objects.
        """
        modalities: Dict[str, AnnData] = {}
        
        # Process each modality
        for mod_key, mod_info in config.data_config.data_info.items():
            if not mod_info.is_single_cell:
                continue
            adata = sc.read_h5ad(mod_info.file_path)
            modalities[mod_key] = adata
        
        # If paired translation, return dict of MuData objects
        if config.paired_translation:
            result = {}
            for mod_key, adata in modalities.items():
                # Create individual MuData for each modality
                mdata = md.MuData({mod_key: adata})
                result[mod_key] = mdata
            print(f"Paired translation: returning {len(result)} MuData objects")
            return result
        
        # For unpaired translation, return a single MuData with all modalities
        mdata = md.MuData(modalities)
        common_cells = list(
            set.intersection(
                *(set(adata.obs_names) for adata in modalities.values())
            )
        )
        print(f"Number of common cells: {len(common_cells)}")
        mdata = mdata[common_cells]
        return mdata