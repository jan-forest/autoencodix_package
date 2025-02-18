from typing import Dict

import numpy as np

from autoencodix.data._datapackage import DataPackage


class DataPackageSplitter:
    def __init__(
        self, data_package: DataPackage, indicies: Dict[str, np.ndarray]
    ) -> None:
        self._data_package = data_package
        self.indicies = indicies

    def _split_data_package(self, indices: np.ndarray) -> DataPackage:
        """
        Split the data package according to the provided indices.

        Parameters
        ----------
        indices : np.ndarray
            Indices to use for splitting the data.

        Returns
        -------
        DataPackage
            A new DataPackage containing the split data.
        """
        if len(indices) == 0:
            return DataPackage()

        split_sc = None
        if self._data_package.multi_sc is not None:
            split_sc = self._data_package.multi_sc[indices]

        split_bulk = None
        if self._data_package.multi_bulk is not None:
            split_bulk = {
                key: df.iloc[indices]
                for key, df in self._data_package.multi_bulk.items()
            }

        split_annotation = None
        if self._data_package.annotation is not None:
            split_annotation = self._data_package.annotation.iloc[indices]

        split_img = None
        if self._data_package.img is not None:
            split_img = [
                img for i, img in enumerate(self._data_package.img) if i in indices
            ]

        return DataPackage(
            multi_sc=split_sc,
            multi_bulk=split_bulk,
            annotation=split_annotation,
            img=split_img,
        )

    def build_datasets(self) -> None:
        """
        Build datasets for training, validation, and testing from a DataPackage.
        Uses pre-aligned IDs to split the data.

        Raises
        ------
        ValueError
            If no data is available for splitting.
        """
        if self._data_package is None:
            raise ValueError("No data package available for splitting")

        # Get split indices based on aligned IDs
        train_data = self._split_data_package(self.indicies["train"])
        valid_data = self._split_data_package(self.indicies["valid"])
        test_data = self._split_data_package(self.indicies["test"])
        return {
            "train": {"data": train_data, "indices": self.indicies["train"]},
            "valid": {"data": valid_data, "indices": self.indicies["valid"]},
            "test": {"data": test_data, "indices": self.indicies["test"]},
        }
