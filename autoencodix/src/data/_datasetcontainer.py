from dataclasses import dataclass
from autoencodix.src.core._base_dataset import BaseDataset

# TODO add tests
@dataclass
class DataSetContainer:
    """
    A container for datasets used in training, validation, and testing.

    Attributes
    ----------
    train : Dataset
        The training dataset.
    valid : Dataset
        The validation dataset.
    test : Dataset
        The testing dataset.
    """

    train: BaseDataset
    valid: BaseDataset
    test: BaseDataset

    def __getitem__(self, key: str) -> BaseDataset:
        """Allows dictionary-like access to datasets."""
        if key not in {"train", "valid", "test"}:
            raise KeyError(f"Invalid key: {key}. Must be 'train', 'valid', or 'test'.")
        return getattr(self, key)

    def __setitem__(self, key: str, value: BaseDataset):
        """Allows dictionary-like assignment of datasets."""
        if key not in {"train", "valid", "test"}:
            raise KeyError(f"Invalid key: {key}. Must be 'train', 'valid', or 'test'.")
        setattr(self, key, value)
