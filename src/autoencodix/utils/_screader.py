import scanpy as sc
from anndata import AnnData

from autoencodix.utils.default_config import DefaultConfig


class SingleCellDataReader:
    @staticmethod
    def read_data(config: DefaultConfig) -> AnnData:
        data_info = config.data_config.data_info
        adatas = {
            k: sc.read_h5ad(v.file_path)
            for k, v in data_info.items()
            if v.is_single_cell
        }

        common_genes = list(
            set.intersection(*(set(adata.var_names) for adata in adatas.values()))
        )

        first_key = next(iter(adatas))
        adata = adatas[first_key][:, common_genes].copy()

        X_set = False
        for k, v in data_info.items():
            if v.is_single_cell:
                sliced_data = adatas[k][:, common_genes]

                if v.is_X:
                    X_set = True
                    adata.X = sliced_data.X

        if not X_set:
            print(f"Warning: No dataset marked as X, using {first_key} as X")

        for k, v in adatas.items():
            sliced_data = v[:, common_genes]
            adata.layers[k] = sliced_data.X

            for col in sliced_data.obs.columns:
                adata.obs[f"{k}_{col}"] = sliced_data.obs[col]

        return adata
