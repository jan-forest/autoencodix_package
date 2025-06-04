from typing import Optional, Union, Type, List
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput


class GeneralTrainer(BaseTrainer):
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
    loss_type : Type[BaseLoss] which loss function to use

    Methods
    -------
    train()
        Train the model standard PyTorch training loop
        Populates the result object with training dynamics and model checkpoints
    _capture_dynamics(epoch, model_output)
        Capture the model dynamics at each epoch (latentspace and reconstruction)

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

    def train(self) -> Result:
        with self._fabric.autocast():
            for epoch in range(self._config.epochs):
                train_outputs = []
                valid_outputs = []
                self._model.train()
                epoch_loss = 0.0
                epoch_sub_losses: defaultdict[str, float] = defaultdict(lambda: 0.0)
                valid_epoch_sub_losses: defaultdict[str, float] = defaultdict(
                    lambda: 0.0
                )
                for _, (features, sample_ids) in enumerate(self._trainloader):
                    self._optimizer.zero_grad()
                    model_outputs = self._model(features)
                    loss, sub_losses = self._loss_fn(
                        model_output=model_outputs, targets=features
                    )
                    self._fabric.backward(loss)
                    self._optimizer.step()
                    epoch_loss += loss.item()
                    train_outputs.append(model_outputs)
                    for k, v in sub_losses.items():
                        epoch_sub_losses[k] += v.item()
                # loss per epoch savving -----------------------------------
                self._result.losses.add(
                    epoch=epoch, split="train", data=epoch_loss / len(self._trainloader)
                )
                self._result.sub_losses.add(
                    epoch=epoch,
                    split="train",
                    data={
                        k: v / len(self._trainloader)
                        for k, v in epoch_sub_losses.items()
                    },
                )

                # to account for shuffling, we save the sample ids for training dynamics visualization
                print(f"sample_ids: {sample_ids}")
                self._result.sample_ids.add(
                    epoch=epoch, split="train", data={"sample_ids": sample_ids}
                )

                # validation loss per epoch ---------------------------------
                if self._validset:
                    self._model.eval()
                    with torch.no_grad():
                        valid_loss = 0.0
                        for _, (features, _) in enumerate(self._validloader):
                            valid_model_outputs = self._model(features)
                            loss, sub_losses = self._loss_fn(
                                model_output=valid_model_outputs, targets=features
                            )
                            valid_loss += loss.item()
                            valid_outputs.append(valid_model_outputs)
                            for k, v in sub_losses.items():
                                valid_epoch_sub_losses[k] += v.item()
                    self._result.losses.add(
                        epoch=epoch,
                        split="valid",
                        data=valid_loss / len(self._validloader),
                    )
                    self._result.sub_losses.add(
                        epoch=epoch,
                        split="valid",
                        data={
                            k: v / len(self._validloader)
                            for k, v in valid_epoch_sub_losses.items()
                        },
                    )

                if not (epoch + 1) % self._config.checkpoint_interval:
                    self._result.model_checkpoints.add(
                        epoch=epoch, data=self._model.state_dict()
                    )
                    print(f"Epoch: {epoch}, Loss: {epoch_loss}")
                    self._capture_dynamics(
                        epoch=epoch, model_output=train_outputs, split="train"
                    )
                    if self._validset:
                        self._capture_dynamics(
                            epoch=epoch,
                            model_output=valid_outputs,
                            split="valid",
                        )

            self._result.model = next(self._model.children())
            return self._result

    def decode(self, x: torch.tensor):
        with self._fabric.autocast(), torch.no_grad():
            x = self._fabric.to_device(obj=x)
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Expected input to be a torch.Tensor, got {type(x)} instead."
                )
            return self._model.decode(x=x)

    def _capture_dynamics(
        self, epoch: int, model_output: List[ModelOutput], split: str
    ):
        # Concatenate tensors from all model outputs
        latentspaces = torch.cat([output.latentspace for output in model_output], dim=0)
        reconstructions = torch.cat(
            [output.reconstruction for output in model_output], dim=0
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
        logvars = [
            output.latent_logvar
            for output in model_output
            if output.latent_logvar is not None
        ]
        if logvars:
            sigmas = torch.cat(
                logvars,
                dim=0,
            )
            self._result.sigmas.add(
                epoch=epoch, split=split, data=sigmas.cpu().detach().numpy()
            )
        mus = [
            output.latent_mean
            for output in model_output
            if output.latent_mean is not None
        ]
        if mus:
            means = torch.cat(
                mus,
                dim=0,
            )
            self._result.mus.add(
                epoch=epoch, split=split, data=means.cpu().detach().numpy()
            )

    def predict(self, data: BaseDataset, model: torch.nn.Module) -> Result:
        """
        Decided to add predict method to the trainer class.
        This violates SRP, but the trainer class has a lot of attributes and methods
        that are needed for prediction. So this way we don't need to write so much duplicate code

        Parameters:
            data: BaseDataset unseen data to run inference on
            model: torch.nn.Module model to run inference with

        Returns:
            self._result: Result object containing the inference results

        """
        model.eval()
        inference_loader = DataLoader(
            data,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.n_workers,
        )
        inference_loader = self._fabric.setup_dataloaders(inference_loader)  # type: ignore
        with self._fabric.autocast(), torch.no_grad():
            outputs = []
            for _, (data, _) in enumerate(inference_loader):
                model_output = model(data)
                outputs.append(model_output)
            self._capture_dynamics(epoch=-1, model_output=outputs, split="test")

        return self._result
