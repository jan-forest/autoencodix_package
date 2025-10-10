from typing import Literal, Dict, Union
from autoencodix.data import MultiModalDataset, NumericDataset, DatasetContainer
import anndata as ad
import pandas as pd


class AnnDataConverter:
    """Utility class for converting datasets into AnnData or multimodal AnnData dictionaries."""

    @staticmethod
    def _numeric_ds_to_adata(ds: NumericDataset) -> Dict[str, ad.AnnData]:
        """Convert a NumericDataset to an AnnData object.

        Args:
            ds: The numeric dataset to convert.

        Returns:
            An AnnData object containing the dataset's data, features, and metadata.
        """
        if not isinstance(ds.metadata, pd.DataFrame):
            raise ValueError(
                f"metadata needs to be pd.DataFrame, got {type(ds.metadata)}"
            )
        return {
            "global": ad.AnnData(
                X=ds.data.detach().cpu().numpy(),
                var=pd.DataFrame(ds.feature_ids),
                obs=ds.metadata,
            )
        }

    @staticmethod
    def _parse_multimodal(mds: MultiModalDataset) -> Dict[str, ad.AnnData]:
        """Convert a MultiModalDataset into a dictionary of AnnData objects.

        Args:
            mds: The multimodal dataset to convert.

        Returns:
            A dictionary mapping modality names to AnnData objects.

        Raises:
            NotImplementedError: If any modality is not a NumericDataset.
        """
        result_dict: Dict[str, ad.AnnData] = {}
        for mod_name, dataset in mds.datasets.items():
            if not isinstance(dataset, NumericDataset):
                raise NotImplementedError(
                    f"Feature Importance is only implemented for NumericDataset, got type: {type(dataset)}"
                )
            result_dict[mod_name] = AnnDataConverter._numeric_ds_to_adata(dataset)
        return result_dict

    @staticmethod
    def dataset_to_adata(
        datasetcontainer: DatasetContainer,
        split: Literal["train", "valid", "test"] = "train",
    ) -> Dict[str, ad.AnnData]:
        """Convert a DatasetContainer split to an AnnData or multimodal AnnData dictionary.

        Args:
            datasetcontainer: Container holding train/valid/test datasets.
            split: The dataset split to convert. Defaults to "train".

        Returns:
            A single AnnData object (for NumericDataset) or a dictionary of AnnData objects (for MultiModalDataset).

        Raises:
            ValueError: If the specified split does not exist in the DatasetContainer.
            NotImplementedError: If the dataset type is not supported.
        """
        if not hasattr(datasetcontainer, split):
            raise ValueError(
                f"Split: {split} not present in DatasetContainer: {datasetcontainer}"
            )

        ds = datasetcontainer[split]

        if isinstance(ds, MultiModalDataset):
            return AnnDataConverter._parse_multimodal(ds)
        elif isinstance(ds, NumericDataset):
            return AnnDataConverter._numeric_ds_to_adata(ds)
        else:
            raise NotImplementedError(
                f"Conversion not implemented for type: {type(ds)}"
            )
