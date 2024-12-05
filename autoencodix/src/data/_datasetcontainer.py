from dataclasses import dataclass
from torch.utils.data import Dataset

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
    train: Dataset
    valid: Dataset
    test: Dataset

    def __getitem__(self, key: str) -> Dataset:
        """Allows dictionary-like access to datasets."""
        if key not in {"train", "valid", "test"}:
            raise KeyError(f"Invalid key: {key}. Must be 'train', 'valid', or 'test'.")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Dataset):
        """Allows dictionary-like assignment of datasets."""
        if key not in {"train", "valid", "test"}:
            raise KeyError(f"Invalid key: {key}. Must be 'train', 'valid', or 'test'.")
        setattr(self, key, value)
