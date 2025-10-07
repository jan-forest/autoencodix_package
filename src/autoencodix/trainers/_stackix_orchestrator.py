import os
import pickle
from typing import Dict, List, Optional, Tuple, Type, Any

import lightning_fabric
import numpy as np
import pandas as pd
import torch

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._multimodal_dataset import MultiModalDataset
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig


class StackixOrchestrator:
    """StackixOrchestrator coordinates the training of multi-modality VAE stacking.

    This orchestrator manages both parallel and sequential training of modality-specific
    autoencoders, followed by creating a concatenated latent space for the final
    stacked model training. It leverages Lightning Fabric's distribution strategies
    for efficient training.

    Attributes:
    _workdir: Directory for saving intermediate models and results
    modality_models: Dictionary of trained models for each modality
    modality_results: Dictionary of training results for each modality
    _modality_latent_dims: Dictionary of latent dimensions for each modality
    concatenated_latent_spaces: Dictionary of concatenated latent spaces by split
    _dataset_type: Class to use for creating datasets
    _fabric: Lightning Fabric instance for distributed operations
    _trainer_class: Class to use for training (dependency injection)
    trainset: Training dataset containing multiple modalities
    validset: Validation dataset containing multiple modalities
    testset: Test dataset containing multiple modalities
    loss_type: Type of loss function to use for training
    model_type: Type of autoencoder model to use for each modality
    config: Configuration parameters for training and model architecture
    stacked_model: The final stacked autoencoder model (initialized later)
    stacked_trainer: Trainer for the stacked model (initialized later)
    concat_idx: Dictionary tracking the start and end indices of each modality in the concatenated latent space
    dropped_indices_map: Dictionary tracking dropped sample indices for each modality during alignment
    reconstruction_shapes: Dictionary storing original shapes of latent spaces for reconstruction
    common_sample_ids: Common sample IDs across all modalities for alignment

    """

    def __init__(
        self,
        trainset: Optional[MultiModalDataset],
        validset: Optional[MultiModalDataset],
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        testset: Optional[MultiModalDataset] = None,
        trainer_type: Type[GeneralTrainer] = GeneralTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        workdir: str = "./stackix_work",
    ):
        """
        Initialize the StackixOrchestrator with datasets and configuration.

        Args:
            trainset: Training dataset containing multiple modalities
            validset: Validation dataset containing multiple modalities
            config: Configuration parameters for training and model architecture
            model_type: Type of autoencoder model to use for each modality
            loss_type: Type of loss function to use for training
            testset: Dataset with test split
            trainer_type: Type to use for training (default is GeneralTrainer)
            dataset_type: Type to use for creating datasets (default is NumericDataset)
            workdir: Directory to save intermediate models and results (default is "./stackix_work")
        """
        self._trainset = trainset
        self._validset = validset
        self._testset = testset
        self._config = config
        self._model_type = model_type
        self._loss_type = loss_type
        self._workdir = workdir
        self._trainer_class = trainer_type
        self._dataset_type = dataset_type

        # Initialize fabric for distributed operations
        strategy = "auto"
        if hasattr(config, "gpu_strategy"):
            strategy = config.gpu_strategy

        self._fabric = lightning_fabric.Fabric(
            accelerator=(
                str(config.device) if hasattr(config, "device") else None
            ),  # ty: ignore[invalid-argument-type]
            devices=config.n_gpus if hasattr(config, "n_gpus") else 1,
            precision=(
                config.float_precision if hasattr(config, "float_precision") else "32"
            ),
            strategy=strategy,
        )

        # Initialize storage for models and results
        self.modality_trainers: Dict[str, BaseAutoencoder] = {}
        self.modality_results: Dict[str, Result] = {}
        self._modality_latent_dims: Dict[str, int] = {}
        self.concatenated_latent_spaces: Dict[str, torch.Tensor] = {}

        # Stacked model and trainer will be created during the process
        self.stacked_model: Optional[BaseAutoencoder] = None
        self.stacked_trainer: Optional[GeneralTrainer] = None
        self.concat_idx: Dict[str, Optional[Tuple[int, int]]] = {}
        self.dropped_indices_map: Dict[str, np.ndarray] = {}
        self.reconstruction_shapes: Dict[str, torch.Size] = {}
        self.common_sample_ids: Optional[pd.Index] = None

        # Create the directory if it doesn't exist
        os.makedirs(name=self._workdir, exist_ok=True)

    def set_testset(self, testset: MultiModalDataset) -> None:
        """Set the test dataset for the orchestrator.

        Args:
            testset: Test dataset containing multiple modalities
        """
        self._testset = testset

    def _train_modality(
        self,
        modality: str,
        modality_dataset: BaseDataset,
        valid_modality_dataset: Optional[BaseDataset] = None,
    ) -> None:
        """Trains a single modality and returns the trained model and result.

        This function is designed to be executed within a Lightning Fabric context.
        It trains an individual modality model and saves the results to disk.

        Args:
            modality: Modality name/identifier
            modality_dataset: Training dataset for this modality
            valid_modality_dataset: Validation dataset for this modality (default is None)

        Returns:
            Trained model and result object
        """
        print(f"Training modality: {modality}")
        result = Result()
        trainer = self._trainer_class(
            trainset=modality_dataset,
            validset=valid_modality_dataset,
            result=result,
            config=self._config,
            model_type=self._model_type,
            loss_type=self._loss_type,
        )

        result = trainer.train()
        self.modality_results[modality] = result
        self.modality_trainers[modality] = trainer

    def _train_distributed(self, keys: List[str]) -> None:
        """Trains modality models in a distributed fashion using Lightning Fabric.

        Uses Lightning Fabric's built-in capabilities for distributing work across devices.
        Each process trains a subset of modalities, then loads results from other processes.

        Args:
            keys: List of modality keys to train
        """
        # For parallel training with multiple devices
        strategy = (
            self._config.gpu_strategy
            if hasattr(self._config, "gpu_strategy")
            else "auto"
        )
        n_gpus = self._config.n_gpus if hasattr(self._config, "n_gpus") else 1
        device: str = self._config.device

        if strategy == "auto" and device in ["cuda", "gpu"]:
            strategy = "ddp" if n_gpus > 1 else "dp"

        # Create a fabric instance with appropriate parallelism strategy
        fabric = lightning_fabric.Fabric(
            accelerator=device,
            devices=min(n_gpus, len(keys)),  # No more devices than modalities
            precision=(
                self._config.float_precision
                if hasattr(self._config, "float_precision")
                else "32"
            ),
            strategy=strategy,
        )

        # Launch distributed training
        fabric.launch()

        # Device ID within the process group
        local_rank = fabric.local_rank if hasattr(fabric, "local_rank") else 0
        world_size = fabric.world_size if hasattr(fabric, "world_size") else 1

        # Distribute modalities across ranks
        my_modalities = [k for i, k in enumerate(keys) if i % world_size == local_rank]

        # Train assigned modalities
        for modality in my_modalities:
            train_dataset = self._trainset.datasets[modality]
            valid_dataset = (
                self._validset.datasets.get(modality)
                if self._validset and hasattr(self._validset, "datasets")
                else None
            )

            self._train_modality(
                modality=modality,
                modality_dataset=train_dataset,
                valid_modality_dataset=valid_dataset,
            )
            self._modality_latent_dims[modality] = train_dataset.get_input_dim()

        # Synchronize across processes to ensure all modalities are trained
        if world_size > 1:
            fabric.barrier()

        # Load models and results from other processes
        for modality in keys:
            if modality in self.modality_trainers.keys():
                continue
            # Load model state dict
            model_path = os.path.join(self._workdir, f"{modality}_model.ckpt")

            # Get input dimension for this modality
            input_dim = self._trainset.datasets[modality].get_input_dim()

            # Initialize a new model
            model = self._model_type(config=self._config, input_dim=input_dim)
            model.load_state_dict(
                state_dict=torch.load(f=model_path, map_location="cpu")
            )

            # Load result
            with open(
                file=os.path.join(self._workdir, f"{modality}_result.pkl"),
                mode="rb",
            ) as f:
                result = pickle.load(file=f)

            self.modality_trainers[modality] = model
            self.modality_results[modality] = result
            self._modality_latent_dims[modality] = input_dim

    def _train_sequential(self, keys: List[str]) -> None:
        """Trains modality models sequentially on a single device.

        Used when distributed training is not available or not necessary.
        Processes each modality one after another.

        Args:
            keys: List of modality keys to train
        """
        for modality in keys:
            print(f"Training modality: {modality}")
            train_dataset = self._trainset.datasets[modality]
            valid_dataset = (
                self._validset.datasets.get(modality) if self._validset else None
            )

            self._train_modality(
                modality=modality,
                modality_dataset=train_dataset,
                valid_modality_dataset=valid_dataset,
            )

            self._modality_latent_dims[modality] = train_dataset.get_input_dim()

    def _extract_latent_spaces(
        self, result_dict: Dict[str, Any], split: str = "train"
    ) -> torch.Tensor:
        """Extracts, aligns, and concatenates latent spaces, populating instance attributes for later reconstruction.

        Args:
            result_dict: Dictionary of Result objects from modality training
            split: Data split to extract latent spaces from ("train", "valid", "test"), default is "train"
        """
        # --- Step 1: Extract Latent Spaces and Sample IDs ---
        all_latents = {}
        all_sample_ids = {}
        for name, result in result_dict.items():
            try:
                latent_space = result.latentspaces.get(split=split, epoch=-1)
                sample_ids = result.sample_ids.get(split=split, epoch=-1)

                if latent_space.shape[0] != len(sample_ids):
                    raise ValueError(
                        f"Mismatch between latent space rows ({latent_space.shape[0]}) and sample IDs ({len(sample_ids)}) for modality '{name}'."
                    )

                all_latents[name] = latent_space
                all_sample_ids[name] = sample_ids
                self.reconstruction_shapes[name] = latent_space.shape
            except Exception as e:
                raise ValueError(
                    f"Failed to extract data for modality {name} on split {split}: {e}"
                )

        if not all_latents:
            raise ValueError("No latent spaces were extracted.")

        # --- Step 2: Find Common Sample IDs ---
        initial_ids = set(next(iter(all_sample_ids.values())))
        common_ids_set = initial_ids.intersection(
            *[set(ids) for ids in all_sample_ids.values()]
        )

        if not common_ids_set:
            raise ValueError(
                "No common samples found across all modalities for concatenation."
            )
        # common_ids = [cid for cid in list(common_ids_set) if cid]
        self.common_sample_ids = pd.Index(sorted(list(common_ids_set)))

        # self.common_sample_ids = pd.Index(sorted(list(common_ids_set)))
        print(
            f"Found {len(self.common_sample_ids)} common samples for the stacked autoencoder."
        )

        # --- Step 3 & 4: Align Latent Spaces and Track Dropped Indices ---
        aligned_latents_list = []
        start_idx = 0
        for name, latent_space in all_latents.items():
            original_ids = pd.Index(all_sample_ids[name])
            latent_df = pd.DataFrame(latent_space, index=original_ids)

            aligned_df = latent_df.loc[self.common_sample_ids]
            aligned_latents_list.append(torch.from_numpy(aligned_df.values))

            end_idx = start_idx + aligned_df.shape[1]
            self.concat_idx[name] = (start_idx, end_idx)
            start_idx = end_idx

            dropped_mask = ~original_ids.isin(self.common_sample_ids)
            self.dropped_indices_map[name] = np.where(dropped_mask)[0]

        # --- Step 5: Concatenate ---
        stackix_input = torch.cat(aligned_latents_list, dim=1)
        return stackix_input

    def train_modalities(self) -> Tuple[Dict[str, BaseAutoencoder], Dict[str, Result]]:
        """Trains all modality-specific models.

        This is the first phase of Stackix training where each modality is
        trained independently before their latent spaces are combined.

        Returns:
            Dictionary of trained models for each modality and Dictionary of training results
            for each modality.

        Raises:
            ValueError: If trainset is not a MultiModalDataset or has no modalities
        """
        # if not isinstance(self._trainset, MulDataset):
        #     raise ValueError("Trainset must be a MultiModalDataset for Stackix training")

        keys = self._trainset.datasets.keys()
        if not keys:
            raise ValueError("No modalities found in trainset")

        n_modalities = len(keys)

        # Determine if we should use distributed training
        n_gpus = self._config.n_gpus if hasattr(self._config, "n_gpus") else 0
        use_distributed = bool(n_gpus and n_gpus > 1 and n_modalities > 1)

        if use_distributed:
            self._train_distributed(keys=keys)
        else:
            self._train_sequential(keys=keys)

        return self.modality_trainers, self.modality_results

    def prepare_latent_datasets(self, split: str) -> NumericDataset:
        """Prepares datasets with concatenated latent spaces for stacked model training.

        This is the second phase of Stackix training where latent spaces from
        all modalities are extracted and concatenated.

        Returns:
            Training and validation datasets with concatenated latent spaces

        Raises:
            ValueError: If no modality models have been trained or no latent spaces could be extracted
        """
        if not self.modality_trainers:
            raise ValueError(
                "No modality models have been trained. Call train_modalities() first."
            )

        # Extract and concatenate latent spaces
        if split == "test":
            if self._testset is None:
                raise ValueError(
                    "No test dataset available. Please provide a test dataset for prediction."
                )
            latent = self._extract_latent_spaces(
                result_dict=self.predict_modalities(data=self._testset), split=split
            )
            ds = NumericDataset(
                data=latent,
                config=self._config,
                sample_ids=self.common_sample_ids,
                feature_ids=[
                    f"{k}_latent_{i}"
                    for k, (start, end) in self.concat_idx.items()
                    for i in range(start, end)
                ],
            )
            return ds

        latent = self._extract_latent_spaces(
            result_dict=self.modality_results, split=split
        )
        # Create datasets for the concatenated latent spaces
        feature_ids = [
            f"{k}_latent_{i}"
            for k, (start, end) in self.concat_idx.items()
            for i in range(start, end)
        ]
        ds = NumericDataset(
            data=latent,
            config=self._config,
            sample_ids=self.common_sample_ids,
            feature_ids=feature_ids,
        )

        return ds

    def predict_modalities(self, data: MultiModalDataset) -> Dict[str, torch.Tensor]:
        """Predicts using the trained models for each modality.

        Args:
            data: Input data for prediction, uses test data if not provided

        Returns:
            Dictionary of reconstructed tensors by modality

        Raises:
            ValueError: If model has not been trained yet or no data is available
        """
        if not self.modality_trainers:
            raise ValueError(
                "No modality models have been trained. Call train_modalities() first."
            )

        predictions = {}
        for modality, trainer in self.modality_trainers.items():
            predictions[modality] = trainer.predict(
                data=data.datasets[modality],
                model=trainer._model,
            )
        return predictions

    def reconstruct_from_stack(
        self, reconstructed_stack: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Reconstructs the full data for each modality from the stacked latent reconstruction.

        Args:
            reconstructed_stack: Tensor with the reconstructed concatenated latent space
        Returns:
            Dictionary of reconstructed tensors by modality
        """
        modality_reconstructions: Dict[str, torch.Tensor] = {}

        for trainer in self.modality_trainers.values():
            trainer._model.eval()

        for name, (start_idx, end_idx) in self.concat_idx.items():
            # =================================================================
            # STEP 1: DE-CONCATENATE THE ALIGNED PART
            # This is the small, dense reconstruction matching the common samples.
            # =================================================================
            aligned_latent_recon = reconstructed_stack[:, start_idx:end_idx]

            # =================================================================
            # STEP 2: RE-ASSEMBLE THE FULL-SIZE LATENT SPACE
            # This is the logic from the old `reconstruct_full_latent` function,
            # now integrated directly here.
            # =================================================================
            original_shape = self.reconstruction_shapes[name]
            dropped_indices = self.dropped_indices_map[name]
            n_original_samples = original_shape[0]

            # Determine the original integer positions of the samples that were KEPT.
            kept_indices = np.delete(np.arange(n_original_samples), dropped_indices)

            # Create a full-size placeholder tensor matching the original latent space.
            # We use zeros, as they are a neutral input for most decoders.
            full_latent_recon = torch.zeros(
                size=original_shape,
                dtype=aligned_latent_recon.dtype,
                device=self._fabric.device,
            )

            # Place the aligned reconstruction data into the correct rows of the full tensor.
            full_latent_recon[kept_indices, :] = self._fabric.to_device(
                aligned_latent_recon
            )

            # =================================================================
            # STEP 3: DECODE THE FULL-SIZE LATENT TENSOR
            # The decoder now receives a tensor with the correct number of samples,
            # including the rows of zeros for the dropped samples.
            # =================================================================
            with torch.no_grad():
                model = self.modality_trainers[name].get_model()
                model = self._fabric.to_device(model)

                final_recon = model.decode(full_latent_recon)
                modality_reconstructions[name] = final_recon.cpu()

        return modality_reconstructions
