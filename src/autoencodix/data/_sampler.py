import torch
from torch.utils.data import Sampler
from typing import Sized, Iterator, List


class BalancedBatchSampler(Sampler[List[int]]):
    """
    A custom PyTorch Sampler that avoids creating a final batch of size 1.

    This sampler behaves like a standard `BatchSampler` but with a key
    difference in handling the last batch. If the last batch would normally
    have a size of 1, this sampler redistributes the last two batches to be
    of roughly equal size. For example, if a dataset of 129 samples is used
    with a batch size of 128, instead of yielding batches of [128, 1], it
    will yield two balanced batches, such as [65, 64].

    This is particularly useful for avoiding issues with layers like
    BatchNorm, which require batch sizes greater than 1, without having to
    drop data (`drop_last=True`).

    Args:
        data_source (Sized): The dataset to sample from.
        batch_size (int): The target number of samples in each batch.
        shuffle (bool): If True, the sampler will shuffle the indices at the
                        start of each epoch.
    """

    def __init__(self, data_source: Sized, batch_size: int, shuffle: bool = True):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer, but got {batch_size}"
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns an iterator over batches of indices.
        """
        n_samples = len(self.data_source)
        if n_samples == 0:
            return

        # Generate a list of indices
        indices = torch.arange(n_samples)
        if self.shuffle:
            # Use a random permutation for shuffling
            indices = torch.randperm(n_samples)

        # Check for the special case where the last batch would be of size 1.
        # This logic only applies if there is more than one batch to begin with.
        if n_samples > self.batch_size and n_samples % self.batch_size == 1:
            # Calculate the number of full batches to yield before special handling
            num_full_batches = n_samples // self.batch_size - 1

            # Yield the full-sized batches
            for i in range(num_full_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                yield indices[start_idx:end_idx].tolist()

            # Handle the last two batches by redistributing them
            remaining_indices_start = num_full_batches * self.batch_size
            remaining_indices = indices[remaining_indices_start:]

            # Split the remaining indices (batch_size + 1) into two roughly equal halves
            split_point = (self.batch_size + 1) // 2
            yield remaining_indices[:split_point].tolist()
            yield remaining_indices[split_point:].tolist()

        else:
            # Standard behavior: yield batches of size `batch_size`
            # The last batch will have size > 1 or there will be no remainder.
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                yield indices[i:end_idx].tolist()

    def __len__(self) -> int:
        """
        Returns the total number of batches in an epoch.
        """
        n_samples = len(self.data_source)
        if n_samples == 0:
            return 0

        # If we are redistributing, we create one extra batch compared to floor division
        if n_samples > self.batch_size and n_samples % self.batch_size == 1:
            return n_samples // self.batch_size + 1
        else:
            # Standard ceiling division to calculate number of batches
            return (n_samples + self.batch_size - 1) // self.batch_size
