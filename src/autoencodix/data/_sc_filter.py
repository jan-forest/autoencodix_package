import pandas as pd
from scipy.sparse import issparse, csr_matrix
from mudata import MuData
from autoencodix.utils.default_config import DataInfo
from autoencodix.data._filter import DataFilter

import scanpy as sc


class SingleCellFilter:
    """
    Preprocessor for MuData single-cell data.
    Handles initial filtering and preprocessing, then applies standard
    DataFilter scaling and filtering, returning updated MuData.
    """

    def __init__(self, mudata: MuData, data_info: DataInfo):
        self.mudata = mudata
        self.data_info = data_info

    def preprocess(self) -> MuData:
        """
        Apply complete preprocessing pipeline:
        1. Single-cell specific filtering and preprocessing
        2. Standard DataFilter filtering and scaling

        Returns:
            Processed MuData object with updated matrices
        """
        print("Applying single-cell preprocessing and standard filtering/scaling")
        mudata_filtered = self.mudata.copy()

        for mod_key, mod_data in mudata_filtered.mod.items():
            print(f"Processing modality: {mod_key}")

            # STEP 1: Single-cell specific preprocessing
            # ------------------------------------------

            # Apply filtering based on config
            print("data infor -------#####")
            print(self.data_info)
            min_genes_count = int(self.data_info.min_genes * mod_data.n_vars)
            min_cells_count = int(self.data_info.min_cells * mod_data.n_obs)

            print(f"Filtering cells with fewer than {min_genes_count} genes")
            sc.pp.filter_cells(mod_data, min_genes=min_genes_count)

            print(f"Filtering genes expressed in fewer than {min_cells_count} cells")
            sc.pp.filter_genes(mod_data, min_cells=min_cells_count)

            # Apply normalization based on config
            if self.data_info.normalize_counts:
                print(f"Normalizing counts in modality {mod_key}")
                sc.pp.normalize_total(mod_data, target_sum=1e4)

            if self.data_info.log_transform:
                print(f"Applying log transformation in modality {mod_key}")
                sc.pp.log1p(mod_data)

            # Process specific layers if requested
            if self.data_info.selected_layers:
                print(
                    f"Using selected layers for modality {mod_key}: {self.data_info.selected_layers}"
                )

                dfs = []
                for layer_name in self.data_info.selected_layers:
                    if layer_name == "X":
                        layer_data = mod_data.X
                    else:
                        if layer_name not in mod_data.layers:
                            print(
                                f"Warning: Layer {layer_name} not found in modality {mod_key}"
                            )
                            continue
                        layer_data = mod_data.layers[layer_name]

                    if issparse(layer_data):
                        temp_df = pd.DataFrame.sparse.from_spmatrix(
                            layer_data,
                            index=mod_data.obs_names,
                            columns=[
                                f"{layer_name}_{gene}" for gene in mod_data.var_names
                            ],
                        )
                    else:
                        temp_df = pd.DataFrame(
                            layer_data,
                            index=mod_data.obs_names,
                            columns=[
                                f"{layer_name}_{gene}" for gene in mod_data.var_names
                            ],
                        )

                    dfs.append(temp_df)

                if dfs:
                    combined_df = pd.concat(dfs, axis=1)

                    # STEP 2: Apply standard DataFilter filtering and scaling
                    # -----------------------------------------------------
                    # Convert sparse to dense if needed
                    if hasattr(combined_df, "sparse"):
                        df_to_process = combined_df.sparse.to_dense()
                    else:
                        df_to_process = combined_df

                    # Apply DataFilter
                    data_filter = DataFilter(df=df_to_process, data_info=self.data_info)
                    filtered_df = data_filter.filter()
                    print(f"Filtered with DataFilter: shape {filtered_df.shape}")
                    scaled_df = data_filter.scale(filtered_df)
                    print(f"Scaled with DataFilter: shape {scaled_df.shape}")

                    # Update MuData with filtered/scaled data
                    if issparse(mod_data.X):
                        mod_data.X = csr_matrix(scaled_df.values)
                    else:
                        mod_data.X = scaled_df.values

                    # Update var_names to match the filtered columns
                    mod_data.var_names = pd.Index(scaled_df.columns)
            else:
                # No layers specified, process the main X matrix
                if issparse(mod_data.X):
                    df = pd.DataFrame.sparse.from_spmatrix(
                        mod_data.X, index=mod_data.obs_names, columns=mod_data.var_names
                    )
                    # Convert to dense for DataFilter
                    if hasattr(df, "sparse"):
                        df = df.sparse.to_dense()
                else:
                    df = pd.DataFrame(
                        mod_data.X, index=mod_data.obs_names, columns=mod_data.var_names
                    )

                # STEP 2: Apply standard DataFilter filtering and scaling
                # -----------------------------------------------------
                data_filter = DataFilter(df=df, data_info=self.data_info)
                filtered_df = data_filter.filter()
                print(f"Filtered with DataFilter: shape {filtered_df.shape}")
                scaled_df = data_filter.scale(filtered_df)
                print(f"Scaled with DataFilter: shape {scaled_df.shape}")

                # Update MuData with filtered/scaled data
                # First subset to match filtered columns
                mod_data._inplace_subset_var(filtered_df.columns)

                # Then update X with the filtered/scaled values
                if issparse(mod_data.X):
                    mod_data.X = csr_matrix(scaled_df.values)
                else:
                    mod_data.X = scaled_df.values

        print(f"Processed MuData with {len(mudata_filtered.mod)} modalities")
        return mudata_filtered
