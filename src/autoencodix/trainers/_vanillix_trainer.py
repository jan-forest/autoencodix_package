from typing import Optional, Union

import torch
import torch.nn.functional as F

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.utils._result import Result
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils.default_config import DefaultConfig

# TODO: write tests
# internal check done
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
        called_from: str,
    ):
        # see _base_trainer.py for more details, handles input validation and reproducibility, sets up Fabric
        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            called_from=called_from,
        )

    def train(self) -> Result:

        for epoch in range(self._config.epochs):
            self._model.train()
            epoch_loss = 0.0
            for _, (features, _) in enumerate(
                self._trainloader
            ):  # features, _ is a tuple of data and label
                # acutal training step --------------------------------------
                print(f"type(features): {type(features)}")
                print(f" features: {features}")
                self._optimizer.zero_grad()
                model_outputs = self._model(features)
                loss = self._loss_fn(model_outputs, features)
                self._fabric.backward(loss)
                self._optimizer.step()
                epoch_loss += loss.item()
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
                        model_output = self._model(features)
                        loss = self._loss_fn(model_output, features)
                        valid_loss += loss.item()
                self._result.losses.add(
                    epoch=epoch, split="valid", data=valid_loss / len(self._validloader)
                )

            # checkpointing -------------------------------------------------------
            if not (epoch + 1) % self._config.checkpoint_interval:
                print(f"Epoch: {epoch}, Loss: {epoch_loss}")
                self._capture_dynamics(epoch=epoch, model_output=model_output)

        # Update results
        self._result.model = self._model

        return self._result

    def _capture_dynamics(self, epoch, model_output):
        self._result.model_checkpoints[epoch] = self._model.state_dict()
        self._result.latentspaces.add(
            epoch=epoch,
            split="train",
            data=model_output.latentspace.cpu().detach().numpy(),
        )
        self._result.reconstructions.add(
            epoch=epoch,
            split="train",
            data=model_output.reconstruction.cpu().detach().numpy(),
        )
        if self._validset:
            self._result.latentspaces.add(
                epoch=epoch,
                split="valid",
                data=model_output.latentspace.cpu().detach().numpy(),
            )
            self._result.reconstructions.add(
                epoch=epoch,
                split="valid",
                data=model_output.reconstruction.cpu().detach().numpy(),
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
