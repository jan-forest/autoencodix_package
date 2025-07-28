import torch
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils.default_config import DefaultConfig

import numpy as np


class MultiModalDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, datasets: Dict[str, BaseDataset], config: DefaultConfig):
        self.datasets = datasets
        self.n_modalities = len(self.datasets.keys())
        self.sample_to_modalities = self._build_sample_map()
        self.sample_ids: List[Any] = list(self.sample_to_modalities.keys())
        self.config = config
        self.data = next(iter(self.datasets.values())).data
        self.feature_ids = None # TODO

        # Build reverse lookup tables once
        self._id_to_idx = {
            mod: {sid: idx for idx, sid in enumerate(ds.sample_ids)}
            for mod, ds in self.datasets.items()
        }
        self.paired_sample_ids = self._get_paired_sample_ids()
        self.unpaired_sample_ids = list(
            set(self.sample_ids) - set(self.paired_sample_ids)
        )

        self.sampler = CoverageEnsuringSampler(self, batch_size=self.config.batch_size)
        self.collate_fn = create_multimodal_collate_fn(self)

    def _build_sample_map(self):
        sample_to_mods = {}
        for modality, dataset in self.datasets.items():
            for sid in dataset.sample_ids:
                sample_to_mods.setdefault(sid, set()).add(modality)
        return sample_to_mods

    def _get_paired_sample_ids(self):
        return [
            sid
            for sid, mods in self.sample_to_modalities.items()
            if all(mod in mods for mod in self.datasets.keys())
        ]

    def __len__(self):
        return len(self.paired_sample_ids)

    def __getitem__(self, sample_id: str):
        """
        Fetch a full multi-modal sample by ID.
        This replaces the collate_fn by returning a single structured sample.
        """
        sample = {"sample_id": sample_id, "dtype": ""}
        for mod, ds in self.datasets.items():
            if sample_id not in self._id_to_idx[mod]:
                sample[mod] = None
                continue
            idx = self._id_to_idx[mod][sample_id]
            _, data, _ = ds[idx]
            sample[mod] = data
        return sample


class CoverageEnsuringSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures all samples are seen at least once per epoch for each modality.
    Uses a two-phase approach: coverage phase + random phase.
    """

    def __init__(
        self, multimodal_dataset: MultiModalDataset, paired_ratio=0.5, batch_size=64
    ):
        self.dataset = multimodal_dataset
        self.paired_ids = multimodal_dataset.paired_sample_ids
        self.unpaired_ids = multimodal_dataset.unpaired_sample_ids
        self.batch_size = batch_size
        self.paired_ratio = paired_ratio

        total_paired = len(self.paired_ids)
        total_unpaired = len(self.unpaired_ids)

        if total_paired == 0:
            self.paired_ratio = 0.0
        elif total_unpaired == 0:
            self.paired_ratio = 1.0
        else:
            # Use requested ratio, but ensure we have enough samples
            max_possible_paired = total_paired / (total_paired + total_unpaired)
            self.paired_ratio = min(paired_ratio, max_possible_paired)
        # Build modality-specific sample lists
        self.modality_samples = {}
        for modality in multimodal_dataset.datasets.keys():
            self.modality_samples[modality] = multimodal_dataset.datasets[
                modality
            ].sample_ids

    def __iter__(self):
        # Phase 1: Coverage phase - ensure all samples are seen at least once
        coverage_batches = self._generate_coverage_batches()

        # Phase 2: Random phase - fill remaining batches randomly
        random_batches = self._generate_random_batches(coverage_batches)

        for batch in coverage_batches + random_batches:
            yield batch

    def _generate_coverage_batches(self):
        """Generate batches that ensure all samples are covered"""
        coverage_batches = []

        covered = {mod: set() for mod in self.modality_samples.keys()}

        while not all(
            len(covered[mod]) == len(self.modality_samples[mod])
            for mod in self.modality_samples.keys()
        ):
            batch = []

            for modality in self.modality_samples.keys():
                uncovered = [
                    s
                    for s in self.modality_samples[modality]
                    if s not in covered[modality]
                ]

                if uncovered:
                    take = min(
                        len(uncovered), self.batch_size // len(self.modality_samples)
                    )
                    selected = np.random.choice(uncovered, size=take, replace=False)
                    batch.extend(selected)
                    covered[modality].update(selected)

            # Fill remaining batch slots with random samples
            while len(batch) < self.batch_size:
                if len(batch) < self.batch_size * self.paired_ratio and self.paired_ids:
                    sample = np.random.choice(self.paired_ids)
                    batch.append(sample)
                elif self.unpaired_ids:
                    sample = np.random.choice(self.unpaired_ids)
                    batch.append(sample)
                else:
                    break

            batch = list(set(batch))
            if len(batch) >= self.batch_size:
                batch = batch[: self.batch_size]

            if batch:
                coverage_batches.append(batch)

        return coverage_batches

    def _generate_random_batches(self, coverage_batches):
        """Generate additional random batches to fill the epoch"""
        total_samples = len(self.paired_ids) + len(self.unpaired_ids)
        covered_samples = sum(len(batch) for batch in coverage_batches)
        remaining_samples = max(0, total_samples - covered_samples)

        random_batches = []
        num_random_batches = remaining_samples // self.batch_size

        for _ in range(num_random_batches):
            batch = []

            # Add paired samples
            paired_needed = int(self.batch_size * self.paired_ratio)
            if paired_needed > 0 and self.paired_ids:
                paired_samples = np.random.choice(
                    self.paired_ids,
                    size=min(paired_needed, len(self.paired_ids)),
                    replace=True,
                )
                batch.extend(paired_samples)

            # Add unpaired samples
            unpaired_needed = self.batch_size - len(batch)
            if unpaired_needed > 0 and self.unpaired_ids:
                unpaired_samples = np.random.choice(
                    self.unpaired_ids,
                    size=min(unpaired_needed, len(self.unpaired_ids)),
                    replace=True,
                )
                batch.extend(unpaired_samples)

            if batch:
                random_batches.append(batch)

        return random_batches

    def __len__(self):
        total_samples = len(self.paired_ids) + len(self.unpaired_ids)
        return max(total_samples // self.batch_size, len(self.modality_samples))


def create_multimodal_collate_fn(multimodal_dataset: MultiModalDataset):
    """
    Factory function to create a collate function with access to the dataset.
    This allows us to get metadata and original indices.

    Args:
        multimodal_dataset: The multimodal dataset
        stack_tensors: If True, attempt to stack tensors into a single tensor
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        if not batch:
            return {}

        result = {}
        modalities = list(multimodal_dataset.datasets.keys())

        for modality in modalities:
            result[modality] = {
                "data": [],
                "metadata": [],
                "sample_ids": [],
                "sampled_index": [],
                "dtype": [],
            }

            dataset = multimodal_dataset.datasets[modality]

            for sample in batch:
                sample_id = sample["sample_id"]

                cur_metadata = None
                if sample.get(modality) is not None:
                    result[modality]["data"].append(sample[modality])
                    result[modality]["sample_ids"].append(sample_id)

                    if sample_id in multimodal_dataset._id_to_idx[modality]:
                        original_idx = multimodal_dataset._id_to_idx[modality][
                            sample_id
                        ]
                        result[modality]["sampled_index"].append(original_idx)
                    else:
                        result[modality]["sampled_index"].append(None)

                    idx = multimodal_dataset._id_to_idx[modality].get(sample_id)
                    cur_metadata = dataset.metadata.iloc[idx]

                    result[modality]["metadata"].append(cur_metadata)

            result[modality]["data"] = torch.stack(result[modality]["data"])
            result[modality]["metadata"] = pd.concat(
                result[modality]["metadata"], axis=1
            ).T
        return result

    return collate_fn
