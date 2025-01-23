from typing import Optional, Union, Type, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.utils._result import Result
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig


# internal check done
# tests done
class VanillixTrainer(BaseTrainer):
    """
    Handles model training process for the Vanilla Autoencoder model.

    Attributes
    ----------
    trainset : BaseDataset
        Training dataset
    validset : BaseDataset

    result : Result
        Result object to store the training results
    config : DefaultConfig
        Configuration object containing model parameters
    called_from : str
        Name of the class that called the trainer

    Methods
    -------
    train()
        Train the model standard PyTorch training loop
        Populates the result object with training dynamics and model checkpoints
    _capture_dynamics(epoch, model_output)
        Capture the model dynamics at each epoch (latentspace and reconstruction)
    _loss_fn(model_output, targets) -> torch.Tensor
        Compute the loss function based on the reconstruction loss specified in the config
        We support Mean Squared Error (MSE) and Binary Cross Entropy (BCE) loss functions

    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[Union[BaseDataset, None]],
        result: Result,
        config: Optional[Union[None, DefaultConfig]],
        model_type: Type[BaseAutoencoder],
    ):
        # see _base_trainer.py for more details, handles input validation and reproducibility, sets up Fabric
        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            model_type=model_type,
        )

    def train(self) -> Result:
        with self._fabric.autocast():
            for epoch in range(self._config.epochs):
                train_outputs = []
                valid_outputs = []
                self._model.train()
                epoch_loss = 0.0
                for i, (features, _) in enumerate(
                    self._trainloader
                ):  # features, _ is a tuple of data and label
                    # acutal training step --------------------------------------
                    self._optimizer.zero_grad()
                    model_outputs = self._model(features)
                    loss = self._loss_fn(model_outputs, features)
                    self._fabric.backward(loss)
                    self._optimizer.step()
                    epoch_loss += loss.item()
                    train_outputs.append(model_outputs)
                # capture epoch loss  --------------------------------------------
                self._result.losses.add(
                    epoch=epoch, split="train", data=epoch_loss / len(self._trainloader)
                )

                # valid loop ------------------------------------------------------
                if self._validset:
                    self._model.eval()
                    with torch.no_grad():
                        valid_loss = 0.0
                        for _, (features, _) in enumerate(self._validloader):
                            valid_model_outputs = self._model(features)
                            loss = self._loss_fn(valid_model_outputs, features)
                            valid_loss += loss.item()
                            valid_outputs.append(valid_model_outputs)
                    self._result.losses.add(
                        epoch=epoch,
                        split="valid",
                        data=valid_loss / len(self._validloader),
                    )

                # checkpointing -------------------------------------------------------
                if not (epoch + 1) % self._config.checkpoint_interval:

                    self._result.model_checkpoints.add(
                        epoch=epoch, data=self._model.state_dict()
                    )
                    print(f"Epoch: {epoch}, Loss: {epoch_loss}")
                    self._capture_dynamics(
                        epoch=epoch, model_outputs=train_outputs, split="train"
                    )
                    if self._validset:
                        self._capture_dynamics(
                            epoch=epoch,
                            model_outputs=valid_model_outputs,
                            split="valid",
                        )

            # Update results
            self._result.model = self._model

            return self._result

    def _capture_dynamics(
        self, epoch: int, model_outputs: List[ModelOutput], split: str
    ):
        # Concatenate tensors from all model outputs
        latentspaces = torch.cat(
            [output.latentspace for output in model_outputs], dim=0
        )
        reconstructions = torch.cat(
            [output.reconstruction for output in model_outputs], dim=0
        )

        self._result.latentspaces.add(
            epoch=epoch,
            split=split,
            data=latentspaces.cpu().detach().numpy(),
        )
        self._result.reconstructions.add(
            epoch=epoch,
            split=split,
            data=reconstructions.cpu().detach().numpy(),
        )

    def _loss_fn(
        self, model_output: ModelOutput, targets: torch.Tensor
    ) -> torch.Tensor:
        if self._config.reconstruction_loss == "mse":
            return F.mse_loss(input=model_output.reconstruction, target=targets)
        elif self._config.reconstruction_loss == "bce":
            return F.binary_cross_entropy_with_logits(
                input=model_output.reconstruction, target=targets
            )
        else:
            raise NotImplementedError(
                f"Loss function {self._config.reconstruction_loss} is not implemented."
            )

    def predict(self, data: BaseDataset, model: torch.nn.Module) -> Result:
        """
        Decided to add predict method to the trainer class.
        This violates SRP, but the trainer class has a lot of attributes and methods
        that are needed for prediction. So this way we don't need to write so much duplicate code

        Parameters:
            data
        """
        model.eval()
        inference_loader = DataLoader(
            data,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.n_workers,
        )
        inference_loader = self._fabric.setup_dataloaders(inference_loader)
        with self._fabric.autocast(), torch.no_grad():
            outputs = []
            for _, (data, _) in enumerate(inference_loader):
                model_output = model(data)
                outputs.append(model_output)
            self._capture_dynamics(epoch=-1, model_outputs=outputs, split="test")

        return self._result
