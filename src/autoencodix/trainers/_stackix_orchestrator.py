import os
import pickle
from typing import Dict, List, Optional, Tuple, Type, Union

import lightning_fabric
import numpy as np
import torch
from torch.utils.data import DataLoader

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig


class StackixOrchestrator:
    """
    StackixOrchestrator coordinates the training of multi-modality VAE stacking.

    This orchestrator manages both parallel and sequential training of modality-specific
    autoencoders, followed by creating a concatenated latent space for the final
    stacked model training. It leverages Lightning Fabric's distribution strategies
    for efficient training.

    Attributes
    ----------
    _workdir : str
        Directory for saving intermediate models and results
    modality_models : Dict[str, BaseAutoencoder]
        Dictionary of trained models for each modality
    modality_results : Dict[str, Result]
        Dictionary of training results for each modality
    _modality_latent_dims : Dict[str, int]
        Dictionary of latent dimensions for each modality
    concatenated_latent_spaces : Dict[str, torch.Tensor]
        Dictionary of concatenated latent spaces by split
    _dataset_type : Type[BaseDataset]
        Class to use for creating datasets
    _fabric : lightning_fabric.Fabric
        Lightning Fabric instance for distributed operations
    _trainer_class : Type[GeneralTrainer]
        Class to use for training (dependency injection)
    """

    def __init__(
        self,
        trainset: Optional[StackixDataset],
        validset: Optional[StackixDataset],
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        testset: Optional[StackixDataset] = None,
        trainer_type: Type[GeneralTrainer] = GeneralTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        workdir: str = "./stackix_work",
    ):
        """
        Initialize the StackixOrchestrator with datasets and configuration.

        Parameters
        ----------
        trainset : Optional[StackixDataset]
            Training dataset containing multiple modalities
        validset : Optional[StackixDataset]
            Validation dataset containing multiple modalities
        config : DefaultConfig
            Configuration parameters for training and model architecture
        model_type : Type[BaseAutoencoder]
            Type of autoencoder model to use for each modality
        loss_type : Type[BaseLoss]
            Type of loss function to use for training
        trainer_class : Type[GeneralTrainer]
            Class to use for training (default is GeneralTrainer)
        dataset_class : Type[BaseDataset], optional
            Class to use for creating datasets (default is NumericDataset)
        workdir : str, optional
            Directory to save intermediate models and results (default is "./stackix_work")
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
            accelerator=config.device if hasattr(config, "device") else None,
            devices=config.n_gpus if hasattr(config, "n_gpus") else 1,
            precision=config.float_precision
            if hasattr(config, "float_precision")
            else "32",
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

        # Create the directory if it doesn't exist
        os.makedirs(name=self._workdir, exist_ok=True)

    def set_testset(self, testset: StackixDataset) -> None:
        """
        Set the test dataset for the orchestrator.

        Parameters
        ----------
        testset : StackixDataset
            Test dataset containing multiple modalities
        """
        self._testset = testset

    def _train_modality(
        self,
        modality: str,
        modality_dataset: BaseDataset,
        valid_modality_dataset: Optional[BaseDataset] = None,
    ) -> None:
        """
        Trains a single modality and returns the trained model and result.

        This function is designed to be executed within a Lightning Fabric context.
        It trains an individual modality model and saves the results to disk.

        Parameters
        ----------
        modality : str
            Modality name/identifier
        modality_dataset : BaseDataset
            Training dataset for this modality
        valid_modality_dataset : Optional[BaseDataset]
            Validation dataset for this modality (default is None)

        Returns
        -------
        Tuple[BaseAutoencoder, Result]
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
        """
        Trains modality models in a distributed fashion using Lightning Fabric.

        Uses Lightning Fabric's built-in capabilities for distributing work across devices.
        Each process trains a subset of modalities, then loads results from other processes.

        Parameters
        ----------
        keys : List[str]
            List of modality keys to train
        """
        # For parallel training with multiple devices
        strategy = (
            self._config.gpu_strategy
            if hasattr(self._config, "gpu_strategy")
            else "auto"
        )
        n_gpus = self._config.n_gpus if hasattr(self._config, "n_gpus") else 1
        device = self._config.device if hasattr(self._config, "device") else None

        if strategy == "auto" and device in ["cuda", "gpu"]:
            strategy = "ddp" if n_gpus > 1 else "dp"

        # Create a fabric instance with appropriate parallelism strategy
        fabric = lightning_fabric.Fabric(
            accelerator=device,
            devices=min(n_gpus, len(keys)),  # No more devices than modalities
            precision=self._config.float_precision
            if hasattr(self._config, "float_precision")
            else "32",
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
            train_dataset = self._trainset.dataset_dict[modality]
            valid_dataset = (
                self._validset.dataset_dict.get(modality)
                if self._validset and hasattr(self._validset, "dataset_dict")
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
            input_dim = self._trainset.dataset_dict[modality].get_input_dim()

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
        """
        Trains modality models sequentially on a single device.

        Used when distributed training is not available or not necessary.
        Processes each modality one after another.

        Parameters
        ----------
        keys : List[str]
            List of modality keys to train
        """
        for modality in keys:
            print(f"Training modality: {modality}")
            train_dataset = self._trainset.dataset_dict[modality]
            valid_dataset = (
                self._validset.dataset_dict.get(modality) if self._validset else None
            )

            self._train_modality(
                modality=modality,
                modality_dataset=train_dataset,
                valid_modality_dataset=valid_dataset,
            )

            self._modality_latent_dims[modality] = train_dataset.get_input_dim()

    def _extract_latent_spaces(self, result_dict, split="train") -> torch.Tensor:
        """
        Used Result object from each trainer to extract latent spaces for each modality.
        Then the indices for concatenation are stored in self._concat_idx and the latent spaces
        are concatenated into a single tensor that serves as input for the stacked model.

        Parameters
        ----------
        keys : List[str]
            List of modality keys to process
        split : str
            Data split to extract latent spaces from (default is "train")

        Returns
        -------
        torch.Tensor
            Concatenated latent spaces for all modalities

        """
        self.concat_idx: Dict[str, Optional[Tuple[int]]] = {
            k: None for k in result_dict.keys()
        }
        start_idx = 0
        latent_spaces: List = []
        for name, result in result_dict.items():
            try:
                latent_space = result.latentspaces.get(split=split, epoch=-1)
                end_idx = start_idx + latent_space.shape[1]
                self.concat_idx[name] = (start_idx, end_idx)
                start_idx = end_idx
                latent_spaces.append(latent_space)
            except Exception as e:
                raise ValueError(
                    f"Failed to extract latent space for modality {name}: {e} and split {split} in last epoch"
                )
        stackix_input = np.concatenate(latent_spaces, axis=1)
        return torch.from_numpy(stackix_input)

    def train_modalities(self) -> Tuple[Dict[str, BaseAutoencoder], Dict[str, Result]]:
        """
        Trains all modality-specific models.

        This is the first phase of Stackix training where each modality is
        trained independently before their latent spaces are combined.

        Returns
        -------
        Tuple[Dict[str, BaseAutoencoder], Dict[str, Result]]
            Dictionary of trained models for each modality and Dictionary of training results
            for each modality.

        Raises
        ------
        ValueError
            If trainset is not a StackixDataset or has no modalities
        """
        if not isinstance(self._trainset, StackixDataset):
            raise ValueError("Trainset must be a StackixDataset for Stackix training")

        keys = self._trainset.modality_keys
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

    def prepare_latent_datasets(
        self, split: str
    ) -> Tuple[BaseDataset, Optional[BaseDataset]]:
        """
        Prepares datasets with concatenated latent spaces for stacked model training.

        This is the second phase of Stackix training where latent spaces from
        all modalities are extracted and concatenated.

        Returns
        -------
        Tuple[BaseDataset, Optional[BaseDataset]]
            Training and validation datasets with concatenated latent spaces

        Raises
        ------
        ValueError
            If no modality models have been trained or no latent spaces could be extracted
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
                sample_ids=self._testset.sample_ids,
                feature_ids=[
                    f"{k}_latent_{i}"
                    for k, (start, end) in self.concat_idx.items()
                    for i in range(start, end)
                ],
                split_indices=self._testset.split_indices,
                metadata=self._testset.metadata,
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
            sample_ids=self._trainset.sample_ids,
            feature_ids=feature_ids,
            split_indices=self._trainset.split_indices,
            metadata=self._trainset.metadata,
        )

        return ds

    def predict_modalities(self, data: StackixDataset) -> Dict[str, torch.Tensor]:
        """
        Predicts using the trained models for each modality.

        Parameters
        ----------
        data : StackixDataset
            Input data for prediction, uses test data if not provided

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of reconstructed tensors by modality

        Raises
        ------
        ValueError
            If model has not been trained yet or no data is available
        """
        if not self.modality_trainers:
            raise ValueError(
                "No modality models have been trained. Call train_modalities() first."
            )

        predictions = {}
        for modality, trainer in self.modality_trainers.items():
            predictions[modality] = trainer.predict(
                data=data.dataset_dict[modality],
                model=trainer._model,
            )
        return predictions
