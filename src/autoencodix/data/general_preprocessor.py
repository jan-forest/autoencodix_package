from typing import Any, Dict, List, Optional, Tuple, Union

import mudata as md  # type: ignore
import numpy as np
import pandas as pd
import torch
from anndata import AnnData  # type: ignore
from scipy.sparse import issparse  # type: ignore
import scipy as sp

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.datapackage import DataPackage
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DataCase, DefaultConfig


class GeneralPreprocessor(BasePreprocessor):
    """Preprocessor for handling multi-modal data.

    Attributes:
        _datapackage_dict: Dictionary holding DataPackage objects for each data split.
        _dataset_container: Container holding processed datasets for each split.
        _reverse_mapping_multi_bulk: Reverse mapping for multi-bulk data reconstruction.
        _reverse_mapping_multi_sc: Reverse mapping for multi-single-cell data reconstruction.

    """

    def __init__(
        self, config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None
    ) -> None:
        """Initialize the preprocessor and empty reverse mappings per split.

        Args:
            config: Pipeline configuration.
            ontologies: Ontology definitions (``Ontix``), used for feature
                filtering.

        (LLM-generated docstring, pending human review.)
        """
        super().__init__(config=config, ontologies=ontologies)
        self._datapackage_dict: Optional[Dict[str, Any]] = None
        self._dataset_container: Optional[DatasetContainer] = None
        # Reverse mappings for reconstruction
        self._reverse_mapping_multi_bulk: Dict[
            str, Dict[str, Tuple[List[int], List[str]]]
        ] = {"train": {}, "test": {}, "valid": {}}
        self._reverse_mapping_multi_sc: Dict[
            str, Dict[str, Tuple[List[int], List[str]]]
        ] = {"train": {}, "test": {}, "valid": {}}

    def _combine_layers(
        self, modality_name: str, modality_data: Any
    ) -> List[np.ndarray]:
        """Collect the selected AnnData layers of one single-cell modality.

        Args:
            modality_name: Name of the modality in the data config.
            modality_data (AnnData): Data of this modality.

        Returns:
            The matrices of the selected layers, in config order ("X" is the main
            matrix).

        (LLM-generated docstring, pending human review.)
        """
        layer_list: List[np.ndarray] = []
        selected_layers = self.config.data_config.data_info[
            modality_name
        ].selected_layers
        for layer_name in selected_layers:
            if layer_name == "X":
                layer_list.append(modality_data.X)
            elif layer_name in modality_data.layers:
                layer_list.append(modality_data.layers[layer_name])
        return layer_list

    def _combine_modality_data(
        self,
        mudata: md.MuData,  # ty: ignore[invalid-type-form]
    ) -> Union[np.ndarray, sp.sparse.spmatrix]:  # ty: ignore[invalid-type-form]
        # Reset single-cell reverse mapping
        """Concatenate all modalities and selected layers of a MuData feature-wise.

        Records which columns belong to which modality and layer, so that
        reconstructions can be split back later (see ``_reverse_multi_sc``).

        Args:
            mudata: Single-cell data with one AnnData per modality.

        Returns:
            Combined matrix (cells x features); sparse if all inputs are sparse.

        (LLM-generated docstring, pending human review.)
        """
        modality_data_list: List[np.ndarray] = []
        start_idx = 0

        for modality_name, modality_data in mudata.mod.items():
            self._reverse_mapping_multi_sc[self._split][modality_name] = {}
            layers = self.config.data_config.data_info[modality_name].selected_layers
            for layer_name in layers:
                if layer_name == "X":
                    n_feats = modality_data.shape[1]
                else:
                    n_feats = modality_data.layers[layer_name].shape[1]

                end_idx = start_idx + n_feats
                feature_ids = modality_data.var_names.tolist()
                self._reverse_mapping_multi_sc[self._split][modality_name][
                    layer_name
                ] = (
                    list(range(start_idx, end_idx)),
                    feature_ids,
                )
                start_idx = end_idx

            combined_layers = self._combine_layers(
                modality_name=modality_name, modality_data=modality_data
            )
            modality_data_list.extend(combined_layers)
        all_sparse = all(issparse(arr) for arr in modality_data_list)
        if all_sparse:
            combined = sp.sparse.hstack(modality_data_list, format="csr")
        else:
            dense_layers = [
                arr.toarray() if issparse(arr) else arr  # ty: ignore
                for arr in modality_data_list
            ]
            combined = np.concatenate(dense_layers, axis=1)

        return combined

    def _create_numeric_dataset(
        self,
        data: Union[np.ndarray, sp.sparse.spmatrix],
        config: DefaultConfig,
        split_ids: np.ndarray,
        metadata: pd.DataFrame,
        ids: List[str],
        feature_ids: List[str],
    ) -> NumericDataset:
        # keep sparse data sparse until batch level in training for memory efficency
        """Build a ``NumericDataset``; sparse input stays sparse until batching.

        Args:
            data: Feature matrix (samples x features).
            config: Pipeline configuration.
            split_ids: Indices of the samples in this split.
            metadata: Sample metadata.
            ids: Sample IDs.
            feature_ids: Feature IDs.

        Returns:
            The dataset for one split.

        (LLM-generated docstring, pending human review.)
        """
        ds = NumericDataset(
            data=data,
            config=config,
            split_indices=split_ids,
            metadata=metadata,
            sample_ids=ids,
            feature_ids=feature_ids,
        )
        return ds

    def _process_data_package(self, data_dict: Dict[str, Any]) -> BaseDataset:
        """Turn the DataPackage of one split into a single numeric dataset.

        Multi-bulk modalities are concatenated column-wise, single-cell modalities
        and layers via ``_combine_modality_data``. Column ranges per modality are
        stored for reverting reconstructions.

        Args:
            data_dict: Dict with the split's DataPackage ("data") and sample
                indices ("indices").

        Returns:
            The combined dataset of this split.

        Raises:
            ValueError: If a multi-bulk entry is not a DataFrame.
            NotImplementedError: If multi-bulk modalities have different numbers
                of samples.

        (LLM-generated docstring, pending human review.)
        """
        data, split_ids = data_dict["data"], data_dict["indices"]
        # MULTI-BULK
        if data.multi_bulk is not None:
            # reset bulk mapping
            metadata = data.annotation
            bulk_dict: Dict[str, pd.DataFrame] = data.multi_bulk

            # Check if all DataFrames have the same number of samples
            sample_counts = {}
            for subkey, df in bulk_dict.items():
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(
                        f"Expected a DataFrame for '{subkey}', got {type(df)}"
                    )
                sample_counts[subkey] = df.shape[0]
                # print(f"cur shape: {subkey}: {df.shape}")

            # Validate all modalities have the same number of samples
            unique_sample_counts = set(sample_counts.values())
            if len(unique_sample_counts) > 1:
                sample_count_str = ", ".join(
                    [f"{k}: {v} samples" for k, v in sample_counts.items()]
                )
                raise NotImplementedError(
                    f"Different sample counts across modalities are not currently supported for Varix and Vanillix"
                    "Set requires_pared=True in config."
                    f"Found: {sample_count_str}. All modalities must have the same number of samples."
                )

            combined_cols: List[str] = []
            start_idx = 0
            for subkey, df in bulk_dict.items():
                n_feats = df.shape[1]
                end_idx = start_idx + n_feats
                self._reverse_mapping_multi_bulk[self._split][subkey] = (
                    list(range(start_idx, end_idx)),
                    df.columns.tolist(),
                )
                combined_cols.extend(df.columns.tolist())
                start_idx = end_idx

            combined_df = pd.concat(bulk_dict.values(), axis=1)
            return self._create_numeric_dataset(
                data=combined_df.values,
                config=self.config,
                split_ids=split_ids,
                metadata=metadata,
                ids=combined_df.index.tolist(),
                feature_ids=combined_cols,
            )
        # MULTI-SINGLE-CELL
        elif data.multi_sc is not None:
            # reset single-cell mapping
            mudata: md.MuData = data.multi_sc.get(  # ty: ignore[invalid-type-form]
                "multi_sc", None
            )  # ty: ignore[invalid-type-form]
            first_mod = next(iter(mudata.mod.values()))
            # for single cell we know, we have a shared metadata
            # so we can use the first modality as reference
            # otherwise when using .obs from mudata, we get
            # unintutive column names with modality name prefix

            if mudata is None:
                raise NotImplementedError(
                    "Unpaired multi Single Cell case not implemented vor Varix and Vanillix, set requires_paired=True in config"
                )
            combined_data = self._combine_modality_data(mudata)

            # collect feature IDs in concatenation order
            feature_ids: List[str] = []
            for layers in self._reverse_mapping_multi_sc[self._split].values():
                for _, fids in layers.values():
                    feature_ids.extend(fids)
            return self._create_numeric_dataset(
                data=combined_data,
                config=self.config,
                split_ids=split_ids,
                metadata=first_mod.obs,
                ids=mudata.obs_names.tolist(),
                feature_ids=feature_ids,
            )
        else:
            raise NotImplementedError(
                "GeneralPreprocessor only handles multi_bulk or multi_sc."
            )

    def preprocess(
        self,
        raw_user_data: Optional[DataPackage] = None,
        predict_new_data: bool = False,
    ) -> DatasetContainer:
        # run common preprocessing

        # self._reverse_mapping_multi_bulk.clear()
        # self._reverse_mapping_multi_sc.clear()

        """Run the full preprocessing and return one dataset per split.

        Reads, filters, scales and splits the data (see
        ``BasePreprocessor._general_preprocess``) and combines all modalities of a
        split into one ``NumericDataset``.

        Args:
            raw_user_data: Input data. If None, data is read as defined in the
                config.
            predict_new_data: If True, treat all data as test data and reuse the
                fitted scalers and feature selection (used by ``predict``).

        Returns:
            Container with "train", "valid" and "test" datasets (None for
            empty splits).

        Raises:
            TypeError: If preprocessing produced no data.

        (LLM-generated docstring, pending human review.)
        """
        self._datapackage_dict = self._general_preprocess(
            raw_user_data=raw_user_data, predict_new_data=predict_new_data
        )
        if self._datapackage_dict is None:
            raise TypeError("Datapackage cannot be None")

        # prepare container
        ds_container: DatasetContainer = DatasetContainer()

        for split in ["train", "test", "valid"]:
            split_data = self._datapackage_dict.get(split)
            self._split = split
            if not split_data or split_data["data"] is None:
                ds_container[split] = None  # type: ignore
                continue
            ds = self._process_data_package(split_data)
            ds_container[split] = ds
        self._dataset_container = ds_container
        return ds_container

    def format_reconstruction(
        self, reconstruction: torch.Tensor, result: Optional[Result] = None
    ) -> DataPackage:
        """Split a combined reconstruction back into the original modalities.

        Args:
            reconstruction: Reconstructed data (samples x combined features).
            result: Unused; kept for interface compatibility.

        Returns:
            DataPackage with DataFrames (multi-bulk) or a MuData (single-cell)
            per modality, plus the sample metadata.

        Raises:
            NotImplementedError: For data cases other than multi-bulk and
                multi-single-cell.

        (LLM-generated docstring, pending human review.)
        """
        self._split = self._match_split(n_samples=reconstruction.shape[0])
        if self.config.data_case == DataCase.MULTI_BULK:
            return self._reverse_multi_bulk(reconstruction)
        elif self.config.data_case == DataCase.MULTI_SINGLE_CELL:
            return self._reverse_multi_sc(reconstruction)
        else:
            raise NotImplementedError(
                f"Reconstruction not implemented for {self.config.data_case}"
            )

    def _match_split(self, n_samples: int) -> str:
        """Match the split based on the number of samples.

        Args:
            n_samples: Number of samples in the reconstruction.

        Returns:
            Name of the first split with the same number of samples.

        Raises:
            ValueError: If no split has this number of samples.
        """
        print(f"n_samples in format recon: {n_samples}")
        for split, split_data in self._datapackage_dict.items():
            print(split)
            data = split_data.get("data")
            if data is None:
                continue
            ref_n = data.get_n_samples()["paired_count"]
            print(f"n_samples from datatpackge: {ref_n}")
            if n_samples == data.get_n_samples()["paired_count"]["paired_count"]:
                return split
        raise ValueError(
            f"Cannot find matching split for {n_samples} samples in the dataset."
        )

    def _reverse_multi_bulk(
        self, reconstruction: Union[np.ndarray, torch.Tensor]
    ) -> DataPackage:
        """Rebuild one DataFrame per multi-bulk modality from a reconstruction.

        Args:
            reconstruction: Reconstructed data of the current split.

        Returns:
            DataPackage with ``multi_bulk`` DataFrames and the split's metadata.

        (LLM-generated docstring, pending human review.)
        """
        data_package = DataPackage(
            multi_bulk={},
            multi_sc=None,
            annotation=None,
            img=None,
            from_modality=None,
            to_modality=None,
        )
        # reconstruct each bulk subkey
        dfs: Dict[str, pd.DataFrame] = {}
        for subkey, (indices, fids) in self._reverse_mapping_multi_bulk[
            self._split
        ].items():
            arr = self._slice_tensor(
                reconstruction=reconstruction,
                indices=indices,
            )
            dfs[subkey] = pd.DataFrame(
                arr,
                columns=fids,
                index=self._dataset_container[self._split].sample_ids,
            )
        data_package.annotation = self._dataset_container[self._split].metadata

        data_package.multi_bulk = dfs
        return data_package

    def _slice_tensor(
        self, reconstruction: Union[np.ndarray, torch.Tensor], indices: List[int]
    ) -> np.ndarray:
        """Select feature columns from a reconstruction as a NumPy array.

        Args:
            reconstruction: Reconstructed data.
            indices: Column indices to select.

        Returns:
            The selected columns.

        Raises:
            TypeError: If the input is neither a tensor nor an array.

        (LLM-generated docstring, pending human review.)
        """
        if isinstance(reconstruction, torch.Tensor):
            arr = reconstruction[:, indices].detach().cpu().numpy()
        elif isinstance(reconstruction, np.ndarray):
            arr = reconstruction[:, indices]
        else:
            raise TypeError(
                f"Expected reconstruction to be a torch.Tensor or np.ndarray, got {type(reconstruction)}"
            )
        return arr

    def _reverse_multi_sc(self, reconstruction: torch.Tensor) -> DataPackage:
        """Rebuild a MuData (one AnnData per modality and layer) from a reconstruction.

        Args:
            reconstruction: Reconstructed data of the current split.

        Returns:
            DataPackage with ``multi_sc={"multi_sc": MuData}`` and the metadata.

        (LLM-generated docstring, pending human review.)
        """
        data_package = DataPackage(
            multi_bulk=None,
            multi_sc=None,
            annotation=None,
            img=None,
            from_modality=None,
            to_modality=None,
        )
        modalities: Dict[str, AnnData] = {}

        for modality_name, layers in self._reverse_mapping_multi_sc[
            self._split
        ].items():
            # rebuild each layer as DataFrame
            layers_dict: Dict[str, pd.DataFrame] = {}
            for layer_name, (indices, fids) in layers.items():
                arr = self._slice_tensor(reconstruction=reconstruction, indices=indices)
                layers_dict[layer_name] = pd.DataFrame(
                    arr,
                    columns=fids,
                    index=self._dataset_container[self._split].sample_ids,
                )

            # extract X layer for AnnData var
            feature_ids = layers.get("X", (None, []))[1]
            var = pd.DataFrame(index=feature_ids)
            X_df = layers_dict.pop("X", None)
            adata = AnnData(
                X=X_df.values if X_df is not None else None,
                obs=self._dataset_container[self._split].metadata,
                var=var,
                layers={k: v.values for k, v in layers_dict.items()},
            )
            modalities[modality_name] = adata

        data_package.multi_sc = {"multi_sc": md.MuData(modalities)}
        data_package.annotation = self._dataset_container[self._split].metadata
        return data_package
