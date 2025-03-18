from typing import Dict, List, Optional, Type
import torch
from torch.utils.data import DataLoader

from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils._model_output import ModelOutput


class StackixTrainer(BaseTrainer):
    """
    Trainer for Stackix architecture that handles training of multiple VAEs and stacking.

    This trainer implements the stacking behavior:
    1. Train individual VAEs for each modality
    2. Extract latent spaces from these VAEs
    3. Concatenate the latent spaces
    4. Train a final VAE on the concatenated space

    Parameters
    ----------
    trainset : BaseDataset
        Training dataset (expected to be a StackixDataset)
    validset : Optional[BaseDataset]
        Validation dataset (expected to be a StackixDataset)
    result : Result
        Result object to store training outcomes
    config : DefaultConfig
        Configuration for training
    model_type : Type[BaseAutoencoder]
        Type of autoencoder model to use for individual modalities and stacked model
    loss_type : Type[BaseLoss]
        Type of loss function to use for training
    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[BaseDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
    ):
        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
        )

        # Attributes to store intermediate models and results
        self.modality_models: Dict[str, BaseAutoencoder] = {}
        self.modality_results: Dict[str, Result] = {}
        self.concatenated_latent_spaces: Dict[str, torch.Tensor] = {}
        self.stacked_model: Optional[BaseAutoencoder] = None

    def train(self) -> Result:
        """
        Execute the complete Stackix training flow.

        This method:
        1. Trains individual VAEs for each modality
        2. Extracts latent spaces from these VAEs
        3. Concatenates the latent spaces
        4. Trains a final VAE on the concatenated space

        Returns
        -------
        Result
            Updated result object containing training outcomes
        """
        print("Starting Stackix training process...")

        # Verify we have a StackixDataset with modalities
        if not hasattr(self._trainset, "data_dict") or not hasattr(
            self._trainset, "modality_keys"
        ):
            raise ValueError(
                "Expected StackixDataset with modalities for Stackix training"
            )

        # Get modality keys
        modality_keys = self._trainset.modality_keys
        print(f"Training on {len(modality_keys)} modalities: {modality_keys}")

        # Step 1: Train individual VAEs for each modality
        print("\nStep 1: Training individual VAEs for each modality")
        self._train_modality_models(modality_keys)

        # Step 2: Extract latent spaces
        print("\nStep 2: Extracting latent spaces from trained VAEs")
        latent_spaces = self._extract_latent_spaces(modality_keys)

        # Step 3: Concatenate latent spaces
        print("\nStep 3: Concatenating latent spaces")
        concat_latent_spaces = self._concatenate_latent_spaces(latent_spaces)
        self.concatenated_latent_spaces = concat_latent_spaces

        # Step 4: Train stacked VAE
        print("\nStep 4: Training final stacked VAE on concatenated latent space")
        self._train_stacked_model(concat_latent_spaces)

        # Set the final model as the primary model for the trainer
        self._model = self.stacked_model

        return self._result

    def _train_modality_models(self, modality_keys: List[str]) -> None:
        """
        Train individual VAE models for each modality.

        Parameters
        ----------
        modality_keys : List[str]
            List of modality names to train models for
        """
        with self._fabric.autocast():
            for modality in modality_keys:
                print(f"Training model for modality: {modality}")

                # Extract tensor data for this modality
                train_tensor = self._trainset.data_dict[modality]
                valid_tensor = (
                    self._validset.data_dict[modality] if self._validset else None
                )

                # Create datasets for this modality
                train_dataset = NumericDataset(
                    data=train_tensor,
                    config=self._config,
                    ids=self._trainset.ids_dict.get(modality, None),
                )

                valid_dataset = None
                if valid_tensor is not None:
                    valid_dataset = NumericDataset(
                        data=valid_tensor,
                        config=self._config,
                        ids=self._validset.ids_dict.get(modality, None),
                    )

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self._config.batch_size,
                    shuffle=True,
                    num_workers=self._config.n_workers,
                )

                valid_loader = None
                if valid_dataset:
                    valid_loader = DataLoader(
                        valid_dataset,
                        batch_size=self._config.batch_size,
                        shuffle=False,
                        num_workers=self._config.n_workers,
                    )

                # Set up Fabric for loaders
                train_loader = self._fabric.setup_dataloaders(train_loader)
                if valid_loader:
                    valid_loader = self._fabric.setup_dataloaders(valid_loader)

                # Create model and optimizer
                input_dim = train_tensor.shape[1]
                model = self._model_type(config=self._config, input_dim=input_dim)
                model = self._fabric.setup_module(model)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self._config.learning_rate,
                    weight_decay=self._config.weight_decay,
                )
                optimizer = self._fabric.setup_optimizers(optimizer)

                # Create loss function
                loss_fn = self._loss_type(config=self._config)

                # Create result container
                modality_result = Result()

                # Train the model
                for epoch in range(self._config.epochs):
                    model.train()
                    epoch_loss = 0.0
                    train_outputs = []

                    for batch_idx, (features, _) in enumerate(train_loader):
                        optimizer.zero_grad()
                        model_outputs = model(features)
                        loss, sub_losses = loss_fn(
                            model_output=model_outputs, targets=features
                        )
                        self._fabric.backward(loss)
                        optimizer.step()

                        epoch_loss += loss.item()
                        train_outputs.append(model_outputs)

                    # Record training loss
                    modality_result.losses.add(
                        epoch=epoch, split="train", data=epoch_loss / len(train_loader)
                    )

                    # Validation step
                    if valid_loader:
                        model.eval()
                        valid_loss = 0.0
                        valid_outputs = []

                        with torch.no_grad():
                            for features, _ in valid_loader:
                                valid_model_outputs = model(features)
                                loss, _ = loss_fn(
                                    model_output=valid_model_outputs, targets=features
                                )
                                valid_loss += loss.item()
                                valid_outputs.append(valid_model_outputs)

                        # Record validation loss
                        modality_result.losses.add(
                            epoch=epoch,
                            split="valid",
                            data=valid_loss / len(valid_loader),
                        )

                    # Store model outputs every checkpoint_interval
                    if (epoch + 1) % self._config.checkpoint_interval == 0:
                        self._capture_modality_dynamics(
                            result=modality_result,
                            epoch=epoch,
                            model_outputs=train_outputs,
                            split="train",
                        )
                        if valid_loader:
                            self._capture_modality_dynamics(
                                result=modality_result,
                                epoch=epoch,
                                model_outputs=valid_outputs,
                                split="valid",
                            )

                # Store trained model and result
                self.modality_models[modality] = model
                self.modality_results[modality] = modality_result

    def _capture_modality_dynamics(
        self, result: Result, epoch: int, model_outputs: List[ModelOutput], split: str
    ) -> None:
        """
        Capture training dynamics for a modality model.

        Parameters
        ----------
        result : Result
            Result object to update
        epoch : int
            Current epoch
        model_outputs : List[ModelOutput]
            List of model outputs
        split : str
            Data split (e.g., "train", "valid")
        """
        # Concatenate tensors from all model outputs
        latentspaces = torch.cat(
            [output.latentspace for output in model_outputs], dim=0
        )
        reconstructions = torch.cat(
            [output.reconstruction for output in model_outputs], dim=0
        )

        result.latentspaces.add(
            epoch=epoch,
            split=split,
            data=latentspaces.cpu().detach().numpy(),
        )

        result.reconstructions.add(
            epoch=epoch,
            split=split,
            data=reconstructions.cpu().detach().numpy(),
        )

        # Handle latent distribution parameters for VAE
        logvars = [
            output.latent_logvar
            for output in model_outputs
            if output.latent_logvar is not None
        ]
        if logvars:
            sigmas = torch.cat(logvars, dim=0)
            result.sigmas.add(
                epoch=epoch, split=split, data=sigmas.cpu().detach().numpy()
            )

        mus = [
            output.latent_mean
            for output in model_outputs
            if output.latent_mean is not None
        ]
        if mus:
            means = torch.cat(mus, dim=0)
            result.mus.add(epoch=epoch, split=split, data=means.cpu().detach().numpy())

    def _extract_latent_spaces(
        self, modality_keys: List[str]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract latent spaces from trained modality models.

        Parameters
        ----------
        modality_keys : List[str]
            List of modality names

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            Dictionary mapping splits to dictionaries of modality latent spaces
        """
        latent_spaces = {"train": {}, "valid": {}, "test": {}}

        # Setup data dict for each split
        splits = {
            "train": self._trainset,
            "valid": self._validset,
            "test": None,  # We'll handle test data in predict()
        }

        with self._fabric.autocast(), torch.no_grad():
            for modality in modality_keys:
                model = self.modality_models[modality]
                model.eval()

                for split_name, dataset in splits.items():
                    if dataset is None:
                        continue

                    # Extract tensor for this modality
                    tensor_data = dataset.data_dict[modality]

                    # Create dataloader
                    temp_dataset = NumericDataset(
                        data=tensor_data,
                        config=self._config,
                        ids=dataset.ids_dict.get(modality, None),
                    )

                    dataloader = DataLoader(
                        temp_dataset,
                        batch_size=self._config.batch_size,
                        shuffle=False,
                        num_workers=self._config.n_workers,
                    )
                    dataloader = self._fabric.setup_dataloaders(dataloader)

                    # Extract latent representations
                    latent_parts = []
                    for features, _ in dataloader:
                        mu, _ = model.encode(features)
                        latent_parts.append(mu)

                    # Concatenate results
                    latent_tensor = torch.cat(latent_parts, dim=0)
                    latent_spaces[split_name][modality] = latent_tensor

        return latent_spaces

    def _concatenate_latent_spaces(
        self, latent_spaces: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate latent spaces from different modalities.

        Parameters
        ----------
        latent_spaces : Dict[str, Dict[str, torch.Tensor]]
            Dictionary mapping splits to dictionaries of modality latent spaces

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping splits to concatenated latent spaces
        """
        concat_spaces = {}

        for split, modality_spaces in latent_spaces.items():
            if not modality_spaces:
                continue

            # Get list of tensors to concatenate
            tensors_to_concat = list(modality_spaces.values())

            # Concatenate along feature dimension (dim=1)
            concatenated = torch.cat(tensors_to_concat, dim=1)
            concat_spaces[split] = concatenated

            print(f"Concatenated latent space for {split}: shape {concatenated.shape}")

        return concat_spaces

    def _train_stacked_model(self, concat_spaces: Dict[str, torch.Tensor]) -> None:
        """
        Train the final VAE on the concatenated latent space.

        Parameters
        ----------
        concat_spaces : Dict[str, torch.Tensor]
            Dictionary mapping splits to concatenated latent spaces
        """
        if "train" not in concat_spaces:
            raise ValueError("No training data in concatenated spaces")

        # Create datasets
        train_tensor = concat_spaces["train"]
        valid_tensor = concat_spaces.get("valid", None)

        train_dataset = NumericDataset(data=train_tensor, config=self._config)
        valid_dataset = (
            NumericDataset(data=valid_tensor, config=self._config)
            if valid_tensor is not None
            else None
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.n_workers,
        )

        valid_loader = None
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self._config.batch_size,
                shuffle=False,
                num_workers=self._config.n_workers,
            )

        # Set up Fabric for loaders
        train_loader = self._fabric.setup_dataloaders(train_loader)
        if valid_loader:
            valid_loader = self._fabric.setup_dataloaders(valid_loader)

        # Create model and optimizer
        input_dim = train_tensor.shape[1]
        model = self._model_type(config=self._config, input_dim=input_dim)
        model = self._fabric.setup_module(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        optimizer = self._fabric.setup_optimizers(optimizer)

        # Create loss function
        loss_fn = self._loss_type(config=self._config)

        # Train the model
        for epoch in range(self._config.epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            train_outputs = []

            for batch_idx, (features, _) in enumerate(train_loader):
                optimizer.zero_grad()
                model_outputs = model(features)
                loss, sub_losses = loss_fn(model_output=model_outputs, targets=features)
                self._fabric.backward(loss)
                optimizer.step()

                epoch_loss += loss.item()
                train_outputs.append(model_outputs)

            # Record training loss
            self._result.losses.add(
                epoch=epoch, split="train", data=epoch_loss / len(train_loader)
            )

            # Record sub-losses
            for loss_name, loss_value in sub_losses.items():
                self._result.sub_losses.add(
                    epoch=epoch, split="train", data={loss_name: loss_value.item()}
                )

            # Validation phase
            if valid_loader:
                model.eval()
                valid_loss = 0.0
                valid_sub_losses = {k: 0.0 for k in sub_losses.keys()}
                valid_outputs = []

                with torch.no_grad():
                    for features, _ in valid_loader:
                        valid_model_outputs = model(features)
                        loss, sub_losses = loss_fn(
                            model_output=valid_model_outputs, targets=features
                        )
                        valid_loss += loss.item()
                        valid_outputs.append(valid_model_outputs)

                        for k, v in sub_losses.items():
                            valid_sub_losses[k] += v.item()

                # Record validation loss
                self._result.losses.add(
                    epoch=epoch, split="valid", data=valid_loss / len(valid_loader)
                )

                # Record validation sub-losses
                self._result.sub_losses.add(
                    epoch=epoch,
                    split="valid",
                    data={
                        k: v / len(valid_loader) for k, v in valid_sub_losses.items()
                    },
                )

            # Store dynamics at checkpoint intervals
            if (epoch + 1) % self._config.checkpoint_interval == 0:
                self._capture_dynamics(
                    epoch=epoch, model_outputs=train_outputs, split="train"
                )

                if valid_loader:
                    self._capture_dynamics(
                        epoch=epoch, model_outputs=valid_outputs, split="valid"
                    )

        # Store the trained stacked model
        self.stacked_model = model

    def predict(
        self, data: BaseDataset, model: Optional[torch.nn.Module] = None
    ) -> Result:
        """
        Run inference with the test data using the stacked model.

        Parameters
        ----------
        data : BaseDataset
            Test dataset
        model : Optional[torch.nn.Module]
            Model to use for prediction (defaults to stacked model)

        Returns
        -------
        Result
            Result object containing prediction outcomes
        """
        if not self.stacked_model and not model:
            raise ValueError(
                "No model available for prediction. Train the model first."
            )

        predict_model = model if model else self.stacked_model

        # If using StackixDataset, we need to first extract latent spaces and concatenate
        if hasattr(data, "data_dict") and hasattr(data, "modality_keys"):
            # We need to extract latent representations from each modality first
            modality_latent_spaces = {}

            with self._fabric.autocast(), torch.no_grad():
                for modality in data.modality_keys:
                    if modality not in self.modality_models:
                        raise ValueError(f"No trained model for modality {modality}")

                    modality_model = self.modality_models[modality]
                    modality_model.eval()

                    # Extract tensor for this modality
                    tensor_data = data.data_dict[modality]

                    # Create dataloader
                    temp_dataset = NumericDataset(
                        data=tensor_data,
                        config=self._config,
                        ids=data.ids_dict.get(modality, None),
                    )

                    dataloader = DataLoader(
                        temp_dataset,
                        batch_size=self._config.batch_size,
                        shuffle=False,
                        num_workers=self._config.n_workers,
                    )
                    dataloader = self._fabric.setup_dataloaders(dataloader)

                    # Extract latent representations
                    latent_parts = []
                    for features, _ in dataloader:
                        mu, _ = modality_model.encode(features)
                        latent_parts.append(mu)

                    # Concatenate results
                    latent_tensor = torch.cat(latent_parts, dim=0)
                    modality_latent_spaces[modality] = latent_tensor

            # Concatenate latent spaces
            tensors_to_concat = list(modality_latent_spaces.values())
            concatenated = torch.cat(tensors_to_concat, dim=1)

            # Create a new dataset with concatenated latent space
            predict_data = NumericDataset(data=concatenated, config=self._config)
        else:
            # If not a StackixDataset, use the data directly
            predict_data = data

        # Now run inference with the stacked model
        predict_loader = DataLoader(
            predict_data,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.n_workers,
        )
        predict_loader = self._fabric.setup_dataloaders(predict_loader)

        predict_model.eval()
        outputs = []

        with self._fabric.autocast(), torch.no_grad():
            for features, _ in predict_loader:
                model_output = predict_model(features)
                outputs.append(model_output)

            # Capture prediction results
            self._capture_dynamics(epoch=-1, model_outputs=outputs, split="test")

        return self._result
