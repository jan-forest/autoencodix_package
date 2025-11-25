from collections import defaultdict

from lightning_fabric.wrappers import _FabricModule
import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset, DataSetTypes
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.data._multimodal_dataset import (
    CoverageEnsuringSampler,
    MultiModalDataset,
    create_multimodal_collate_fn,
)
from autoencodix.modeling._classifier import Classifier
from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.utils._losses import VarixLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._internals import model_trainer_map
from autoencodix.utils._utils import find_translation_keys


class XModalTrainer(BaseTrainer):
    """Trainer for cross-modal autoencoders, implements multimodal training with adversarial component.

    Attributes:
        _trainset: The dataset used for training, must be a MultiModalDataset.
        _n_train_modalities: Number of modalities in the training dataset.
        model_map: Mapping from DataSetTypes to specific autoencoder architectures.
        model_trainer_map: Mapping from autoencoder architectures to their corresponding trainer classes.
        _n_cpus: Number of CPU cores available for data loading.
        latent_dim: Dimensionality of the shared latent space.
        n_test: Number of samples in the test set, if provided.
        n_train: Number of samples in the training set.
        n_valid: Number of samples in the validation set, if provided.
        n_features: Total number of features across all modalities.
        _cur_epoch: Current epoch number during training.
        _is_checkpoint_epoch: Flag indicating if the current epoch is a checkpoint epoch.
        _sub_loss_type: Loss function type for individual modality autoencoders.
        sub_loss: Instantiated loss function for individual modality autoencoders.
        _clf_epoch_loss: Cumulative classifier loss for the current epoch.
        _epoch_loss: Cumulative total loss for the current epoch.
        _epoch_loss_valid: Cumulative total validation loss for the current epoch.
        _modality_dynamics: Dictionary holding model, optimizer, and training state for each modality.
        _latent_clf: Classifier model for adversarial training on latent spaces.
        _clf_optim: Optimizer for the classifier model.
        _clf_loss_fn: Loss function for the classifier.
        _trainloader: DataLoader for the training dataset.
        _validloader: DataLoader for the validation dataset, if provided.
        _model: The instantiated stacked model architecture.
        _validset: The dataset used for validation, if provided, must be a MultiModalDataset.
        _fabric: Lightning Fabric wrapper for device and precision management.

    """

    def __init__(
        self,
        trainset: MultiModalDataset,
        validset: MultiModalDataset,
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        sub_loss_type: Type[BaseLoss] = VarixLoss,
        model_map: Dict[DataSetTypes, Type[BaseAutoencoder]] = {
            DataSetTypes.NUM: VarixArchitecture,
            DataSetTypes.IMG: ImageVAEArchitecture,
        },
        ontologies: Optional[Union[Tuple, List]] = None,
        **kwargs,
    ):
        """Initializes the XModalTrainer with datasets and configuration.

        Args:
            trainset: Training dataset containing multiple modalities
            validset: Validation dataset containing multiple modalities
            result: Result object to store training outcomes
            config: Configuration parameters for training and model architecture
            model_type: Type of autoencoder model to use for each modality (not used directly)
            loss_type: Type of loss function to use for training the stacked model
            sub_loss_type: Type of loss function to use for individual modality autoencoders
            model_map: Mapping from DataSetTypes to specific autoencoder architectures
            ontologies: Ontology information, for compatibility with Ontix
        """
        self._trainset = trainset
        self._n_train_modalities = self._trainset.n_modalities  # type: ignore
        self.model_map = model_map
        self.model_trainer_map = model_trainer_map
        self._n_cpus = os.cpu_count()
        if self._n_cpus is None:
            self._n_cpus = 0

        super().__init__(
            trainset, validset, result, config, model_type, loss_type, ontologies
        )
        # init attributes ------------------------------------------------
        self.latent_dim = config.latent_dim
        self.n_test: Optional[int] = None
        self.n_train = len(trainset.data) if trainset else 0
        self.n_valid = len(validset.data) if validset else 0
        self.n_features = trainset.get_input_dim() if trainset else 0
        self._cur_epoch: int = 0
        self._is_checkpoint_epoch: Optional[bool] = None
        self._sub_loss_type = sub_loss_type
        self.sub_loss = sub_loss_type(config=self._config)
        self._clf_epoch_loss: float = 0.0
        self._epoch_loss: float = 0.0
        self._epoch_loss_valid: float = 0.0
        self._all_dynamics: Dict[str, Dict[int, Dict]] = {
            "test": defaultdict(dict),
            "train": defaultdict(dict),
            "valid": defaultdict(dict),
        }

    def _init_model_architecture(self, ontologies: Optional[Tuple] = None):
        """Override parent's model init - we don't use a single model, needs to be there because is abstract in parent."""
        pass

    def _setup_fabric(self, old_model=None):
        """Sets up the models, optimizers, and data loaders with Lightning Fabric."""

        self._fabric.launch()
        self._init_adversarial_training()
        self._init_modality_training()
        self._latent_clf, self._clf_optim = self._fabric.setup(
            self._latent_clf, self._clf_optim
        )

        self._trainloader = self._fabric.setup_dataloaders(self._trainloader)
        if self._validloader is not None:
            self._validloader = self._fabric.setup_dataloaders(self._validloader)

        for dynamics in self._modality_dynamics.values():
            dynamics["model"], dynamics["optim"] = self._fabric.setup(
                dynamics["model"], dynamics["optim"]
            )

    # def _init_loaders(self):
    #     """Initializes DataLoaders for training and validation datasets."""
    #     trainsampler = CoverageEnsuringSampler(
    #         datasets=self._trainset, batch_size=self._config.batch_size
    #     )
    #     validsampler = CoverageEnsuringSampler(
    #         datasets=self._validset, batch_size=self._config.batch_size
    #     )
    #     collate_fn = create_multimodal_collate_fn(datasets=self._trainset)
    #     valid_collate_fn = create_multimodal_collate_fn(
    #         datasets=self._validset
    #     )
    #     # drop_last handled in custom sampler
    #     self._trainloader = DataLoader(
    #         self._trainset,
    #         batch_sampler=trainsampler,
    #         collate_fn=collate_fn,
    #     )
    #     self._validloader = DataLoader(
    #         self._validset,
    #         batch_sampler=validsampler,
    #         collate_fn=valid_collate_fn,
    #     )

    def _init_loaders(self):
        """Initializes DataLoaders with smart sampler selection based on pairing."""

        def _build_loader(dataset: MultiModalDataset, is_train: bool):
            batch_size = self._config.batch_size
            collate_fn = create_multimodal_collate_fn(multimodal_dataset=dataset)

            if dataset.is_fully_paired:
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=is_train,
                    drop_last=is_train,
                    collate_fn=collate_fn,
                )
            else:
                print(
                    f"Dataset has UNPAIRED samples → using CoverageEnsuringSampler "
                    f"({len(dataset.paired_sample_ids)} paired + {len(dataset.unpaired_sample_ids)} unpaired)"
                )
                sampler = CoverageEnsuringSampler(
                    multimodal_dataset=dataset,
                    batch_size=batch_size,
                )
                return DataLoader(
                    dataset,
                    batch_sampler=sampler,  # note: batch_sampler, not sampler
                    collate_fn=collate_fn,
                )

        # Build train and validation loaders
        self._trainloader = _build_loader(self._trainset, is_train=True)
        self._validloader = _build_loader(self._validset, is_train=False)

        # Optional: expose for logging
        self._train_is_fully_paired = len(self._trainset.unpaired_sample_ids) == 0

    def _init_modality_training(self):
        """Initializes models, optimizers, and training state for each modality."""
        self._modality_dynamics = {
            mod_name: None for mod_name in self._trainset.datasets.keys()
        }

        self._result.sub_results = {
            mod_name: None for mod_name in self._trainset.datasets.keys()
        }

        data_info = self._config.data_config.data_info
        for mod_name, ds in self._trainset.datasets.items():
            simple_name = mod_name.split(".")[1]
            local_epochs = data_info[simple_name].pretrain_epochs

            pretrain_epochs = (
                local_epochs
                if local_epochs is not None
                else self._config.pretrain_epochs
            )

            model_type = self.model_map.get(ds.mytype)
            if model_type is None:
                raise ValueError(
                    f"No Mapping exists for {ds.mytype}, you passed this mapping: {self.model_map}"
                )
            model = model_type(config=self._config, input_dim=ds.get_input_dim())
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )
            self._modality_dynamics[mod_name] = {
                "model": model,
                "optim": optimizer,
                "mp": [],
                "losses": [],
                "config_name": simple_name,
                "pretrain_epochs": pretrain_epochs,
                "mytype": ds.mytype,
                "pretrain_result": None,
            }

    def _init_adversarial_training(self):
        """Initializes the classifier and its optimizer for adversarial training."""
        self._latent_clf = Classifier(
            input_dim=self._config.latent_dim, n_modalities=self._n_train_modalities
        )
        self._clf_optim = torch.optim.AdamW(
            params=self._latent_clf.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        self._clf_loss_fn = torch.nn.CrossEntropyLoss(
            reduction=self._config.loss_reduction
        )

    def _modalities_forward(self, batch: Dict[str, Dict[str, Any]]):
        """Performs forward pass for each modality in the batch and computes losses."""
        for k, v in batch.items():
            model = self._modality_dynamics[k]["model"]
            data = v["data"]

            mp = model(data)

            loss, loss_stats = self.sub_loss(
                model_output=mp, targets=data, epoch=self._cur_epoch
            )

            self._modality_dynamics[k]["loss_stats"] = loss_stats
            self._modality_dynamics[k]["loss"] = loss
            self._modality_dynamics[k]["mp"] = mp

    def _prep_adver_training(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepares concatenated latent spaces and corresponding labels for adversarial training.

        Returns:
            A tuple containing:
            - latents: Concatenated latent space tensor of shape (total_samples, latent_dim)
            - labels: Tensor of modality labels of shape (total_samples,)
        """
        all_latents: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        mod2idx = {
            mod_name: idx for idx, mod_name in enumerate(self._modality_dynamics.keys())
        }

        for mod_name, helper in self._modality_dynamics.items():
            output: ModelOutput = helper["mp"]
            latents = output.latentspace
            label_id = mod2idx[mod_name]
            all_latents.append(latents)
            all_labels.append(
                torch.full(
                    (latents.size(0),),
                    fill_value=label_id,
                    dtype=torch.long,
                    device=self._fabric.device,
                )
            )

        return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)

    def _train_clf(self, latents: torch.Tensor, labels: torch.Tensor) -> None:
        """Performs a single optimization step for the classifier.
        Args:
            latents: Concatenated latent space tensor of shape (total_samples, latent_dim)
            labels: Tensor of modality labels of shape (total_samples,)

        """
        self._clf_optim.zero_grad()
        # We must detach the latents from the computation graph. This serves two purposes:
        # 1. It isolates the classifier, ensuring that gradients only update its weights.
        # 2. It prevents a "two backward passes" error by leaving the graph to the
        #    autoencoders untouched, ready for use in the next stage.
        detached_latents = latents.detach()
        clf_scores = self._latent_clf(detached_latents)
        clf_loss = self._clf_loss_fn(clf_scores, labels)
        self._fabric.backward(clf_loss)
        self._clf_optim.step()
        self._clf_epoch_loss += clf_loss.item()

    def _validate_one_epoch(self) -> Tuple[List[Dict], Dict[str, float], int]:
        """Runs a single, combined validation pass for the entire model, including dedicated metrics for the classifier.

        Returns:
            A tuple containing:
            - epoch_dynamics: List of dictionaries capturing dynamics for each batch
            - sub_losses: Dictionary of accumulated sub-losses for the epoch
            - n_samples_total: Total number of samples processed in validation
        """
        for dynamics in self._modality_dynamics.values():
            dynamics["model"].eval()
        self._latent_clf.eval()

        self._epoch_loss_valid = 0.0
        total_clf_loss = 0.0
        n_samples_total: int = 0
        epoch_dynamics: List[Dict] = []
        sub_losses: Dict[str, float] = defaultdict(float)

        with torch.no_grad():
            for batch in self._validloader:
                with self._fabric.autocast():
                    self._modalities_forward(batch=batch)
                    latents, labels = self._prep_adver_training()
                    n_samples_total += len(labels)
                    clf_scores = self._latent_clf(latents)

                    batch_loss, loss_dict = self._loss_fn(
                        batch=batch,
                        modality_dynamics=self._modality_dynamics,
                        clf_scores=clf_scores,
                        labels=labels,
                        clf_loss_fn=self._clf_loss_fn,
                        is_training=False,
                    )
                    self._epoch_loss_valid += batch_loss.item()
                    for k, v in loss_dict.items():
                        value = v.item() if hasattr(v, "item") else v
                        if "_factor" not in k:
                            sub_losses[k] += value
                        else:
                            sub_losses[k] = value

                    clf_loss = self._clf_loss_fn(clf_scores, labels)
                    total_clf_loss += clf_loss.item()
                    if self._is_checkpoint_epoch:
                        batch_capture = self._capture_dynamics(batch)
                        epoch_dynamics.append(batch_capture)

        sub_losses["clf_loss"] = total_clf_loss
        n_samples_total /= self._n_train_modalities  # n_modalities because each sample is counted once per modality and n_modalites is the same for train and valid
        for k, v in sub_losses.items():
            if "_factor" not in k:
                sub_losses[k] = v / n_samples_total  # Average over all samples
        self._epoch_loss_valid = (
            self._epoch_loss_valid / n_samples_total
        )  # Average over all samples
        return epoch_dynamics, sub_losses, len(self._validset)

    def _train_one_epoch(self) -> Tuple[List[Dict], Dict[str, float], int]:
        """Runs the training loop with corrected adversarial logic.
        Returns:
            A tuple containing:
            - epoch_dynamics: List of dictionaries capturing dynamics for each batch
            - sub_losses: Dictionary of accumulated sub-losses for the epoch
            - n_samples_total: Total number of samples processed in training

        """
        for dynamics in self._modality_dynamics.values():
            dynamics["model"].train()
        self._latent_clf.train()

        self._clf_epoch_loss = 0
        self._epoch_loss = 0
        epoch_dynamics: List[Dict] = []
        sub_losses: Dict[str, float] = defaultdict(float)
        n_samples_total: int = 0  # because of unpaired training we need to sum the samples instead of using len(dataset)

        for batch in self._trainloader:
            with self._fabric.autocast():
                # --- Stage 1: forward for each data modality ---
                self._modalities_forward(batch=batch)

                # --- Stage 2: Train the Classifier ---
                latents, labels = self._prep_adver_training()
                n_samples_total += latents.size(0)
                self._train_clf(latents=latents, labels=labels)

                # --- Stage 3: Train the Autoencoders ---
                for _, dynamics in self._modality_dynamics.items():
                    dynamics["optim"].zero_grad()

                # We re-calculate scores here and cannot reuse the `clf_scores` from Stage 2.
                # The previous scores were from detached latents and would block the gradients
                # needed to train the autoencoders adversarially.
                clf_scores_for_adv = self._latent_clf(latents)

                batch_loss, loss_dict = self._loss_fn(
                    batch=batch,
                    modality_dynamics=self._modality_dynamics,
                    clf_scores=clf_scores_for_adv,
                    labels=labels,
                    clf_loss_fn=self._clf_loss_fn,
                    is_training=True,
                )
            self._fabric.backward(batch_loss)
            for _, dynamics in self._modality_dynamics.items():
                dynamics["optim"].step()

            # --- Logging and Capturing ---
            self._epoch_loss += batch_loss.item()
            for k, v in loss_dict.items():
                value_to_add = v.item() if hasattr(v, "item") else v
                if "_factor" not in k:
                    sub_losses[k] += value_to_add
                else:
                    sub_losses[k] = value_to_add

            if self._is_checkpoint_epoch:
                batch_capture = self._capture_dynamics(batch)
                epoch_dynamics.append(batch_capture)
        n_samples_total /= self._n_train_modalities
        sub_losses["clf_loss"] = self._clf_epoch_loss
        for k, v in sub_losses.items():
            if "_factor" not in k:
                sub_losses[k] = v / n_samples_total  # Average over all samples
        self._epoch_loss = (
            self._epoch_loss / n_samples_total
        )  # Average over all samples

        return epoch_dynamics, sub_losses, n_samples_total

    def _pretraining(self):
        """Pretrain each modality's model if needed."""
        for mod_name, dynamic in self._modality_dynamics.items():
            mytype = dynamic.get("mytype")
            pretrain_epochs = dynamic.get("pretrain_epochs")
            print(f"Check if we need to pretrain: {mod_name}")
            print(f"pretrain epochs : {pretrain_epochs}")
            if not pretrain_epochs:
                print(f"No pretraining for {mod_name}")
                continue
            model_type = self.model_map.get(mytype)
            pretrainer_type = self.model_trainer_map.get(model_type)
            print(f"Starting Pretraining for: {mod_name} with {pretrainer_type}")
            trainset = self._trainset.datasets.get(mod_name)
            validset = self._validset.datasets.get(mod_name)
            pretrainer = pretrainer_type(
                trainset=trainset,
                validset=validset,
                result=Result(),
                config=self._config,
                model_type=model_type,
                loss_type=self._sub_loss_type,
                ontologies=self.ontologies,
            )
            pretrain_result = pretrainer.train(epochs_overwrite=pretrain_epochs)
            self._result.sub_results[f"pretrain.{mod_name}"] = pretrain_result
            self._modality_dynamics[mod_name]["pretrain_result"] = pretrain_result
            self._modality_dynamics[mod_name]["model"] = pretrain_result.model

    def train(self):
        """Orchestrates the full training process for the cross-modal autoencoder."""
        self._pretraining()
        for epoch in range(self._config.epochs):
            self._cur_epoch = epoch
            self._is_checkpoint_epoch = self._should_checkpoint(epoch=epoch)
            self._fabric.print(f"--- Epoch {epoch + 1}/{self._config.epochs} ---")
            train_epoch_dynamics, train_sub_losses, n_samples_train = (
                self._train_one_epoch()
            )

            self._log_losses(
                split="train",
                total_loss=self._epoch_loss,
                sub_losses=train_sub_losses,
                n_samples=n_samples_train,
            )

            if self._validset:
                valid_epoch_dynamics, valid_sub_losses, n_samples_valid = (
                    self._validate_one_epoch()
                )
                self._log_losses(
                    split="valid",
                    total_loss=self._epoch_loss_valid,
                    sub_losses=valid_sub_losses,
                    n_samples=n_samples_valid,
                )
            if self._is_checkpoint_epoch:
                self._fabric.print(f"Storing checkpoint for epoch {epoch}...")
                self._store_checkpoint(
                    split="train", epoch_dynamics=train_epoch_dynamics
                )
                if self._validset:
                    self._store_checkpoint(
                        split="valid",
                        epoch_dynamics=valid_epoch_dynamics,
                    )
        # save final model
        self._store_checkpoint(split="train", epoch_dynamics=train_epoch_dynamics)
        self._store_final_models()
        return self._result

    def decode(self, x: torch.Tensor):
        """Decodes input latent representations
        Args:
            x: Latent representations to decode, shape (n_samples, latent_dim)
        """
        raise NotImplementedError("General decode step is not implemented for XModalix")

    def predict(
        self,
        data: BaseDataset,
        model: Optional[Dict[str, torch.nn.Module]] = None,
        **kwargs,
    ) -> Result:
        """Performs cross-modal prediction from a specified 'from' modality to a 'to' modality.

        The direction is determined by the 'translate_direction' attribute in the config.
        Results are stored in the Result object under split='test' and epoch=-1.

        Args:
            data: A MultiModalDataset containing the input data for the 'from' modality.

        Returns:
            The Result object populated with prediction results.
        """
        split: str = kwargs.pop("split", "test")
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"split must be one of 'train', 'valid', or 'test', got {split}"
            )
        if not isinstance(data, MultiModalDataset):
            raise TypeError(
                f"type of data has to be MultiModalDataset, got: {type(data)}"
            )
        if model is not None and not hasattr(self, "_modality_dynamics"):
            self._modality_dynamics = {mod_name: {} for mod_name in model.keys()}

            for mod_name, mod_model in model.items():
                mod_model.eval()
                if not isinstance(mod_model, _FabricModule):
                    mod_model = self._fabric.setup_module(mod_model)
                mod_model.to(self._fabric.device)
                self._modality_dynamics[mod_name]["model"] = mod_model
            # self._setup_fabric(old_model=model)

        from_key = kwargs.pop("from_key", None)
        to_key = kwargs.pop("to_key", None)
        predict_keys = find_translation_keys(
            config=self._config,
            trained_modalities=list(self._modality_dynamics.keys()),
            from_key=from_key,
            to_key=to_key,
        )
        from_key, to_key = predict_keys["from"], predict_keys["to"]
        from_modality = self._modality_dynamics[from_key]
        to_modality = self._modality_dynamics[to_key]
        from_model = from_modality["model"]
        to_model = to_modality["model"]
        from_model.eval(), to_model.eval()
        # drop_last handled in custom sampler
        inference_loader = DataLoader(
            data,
            batch_sampler=CoverageEnsuringSampler(
                multimodal_dataset=data, batch_size=self._config.batch_size
            ),
            collate_fn=create_multimodal_collate_fn(multimodal_dataset=data),
        )
        inference_loader = self._fabric.setup_dataloaders(inference_loader)  # type: ignore
        epoch_dynamics: List[Dict] = []
        with (
            self._fabric.autocast(),
            torch.inference_mode(),
        ):
            for batch in inference_loader:
                # needed for visualize later
                self._get_vis_dynamics(batch=batch)
                from_z = self._modality_dynamics[from_key]["mp"].latentspace
                translated = to_model.decode(x=from_z)

                # for reference
                to_z = self._modality_dynamics[to_key]["mp"].latentspace
                to_to_reference = to_model.decode(x=to_z)

                batch_capture = self._capture_dynamics(batch)
                translation_key = "translation"

                reference_key = f"reference_{to_key}_to_{to_key}"
                batch_capture["reconstructions"][translation_key] = (
                    translated.cpu().numpy()
                )

                batch_capture["reconstructions"][reference_key] = (
                    to_to_reference.cpu().numpy()
                )

                if "sample_ids" in batch[from_key]:
                    batch_capture["sample_ids"][translation_key] = np.array(
                        batch[from_key]["sample_ids"]
                    )
                if "sample_ids" in batch[to_key]:
                    batch_capture["sample_ids"][reference_key] = np.array(
                        batch[to_key]["sample_ids"]
                    )

                epoch_dynamics.append(batch_capture)

        self._dynamics_to_result(split=split, epoch_dynamics=epoch_dynamics)

        print("Prediction complete.")
        return self._result

    def _get_vis_dynamics(self, batch: Dict[str, Dict[str, Any]]):
        """Runs a forward pass for each modality in the batch and stores the model outputs.

        This is used to capture dynamics for visualization or analysis not for actual translation as specified in predict.
        Args:
            batch: A dictionary where keys are modality names and values are dictionaries containing 'data' tensors.
        """
        for mod_name, mod_data in batch.items():
            model = self._modality_dynamics[mod_name]["model"]
            # Store model output (containing latents, recons, etc.)
            self._modality_dynamics[mod_name]["mp"] = model(mod_data["data"])

    def _capture_dynamics(
        self,
        batch_data: Union[Dict[str, Dict[str, Any]], Any],
    ) -> Union[Dict[str, Dict[str, np.ndarray]], Any]:
        """Captures and returns the dynamics (latents, reconstructions, etc.) for each modality in the batch.

        Args:
            batch_data: A dictionary where keys are modality names and values are dictionaries containing 'data' tensors
                         and optionally 'sample_ids'.
        Returns:
            A dictionary with keys 'latentspaces', 'reconstructions', 'mus', 'sigmas', and 'sample_ids',
            each mapping to another dictionary where keys are modality names and values are numpy arrays.
        """
        captured_data: Dict[str, Dict[str, Any]] = {
            "latentspaces": {},
            "reconstructions": {},
            "mus": {},
            "sigmas": {},
            "sample_ids": {},
        }
        for mod_name, dynamics in self._modality_dynamics.items():
            sample_ids = batch_data[mod_name].get("sample_ids")
            # get index of
            if sample_ids is not None:
                captured_data["sample_ids"][mod_name] = np.array(sample_ids)

            model_output = dynamics["mp"]
            captured_data["latentspaces"][mod_name] = (
                model_output.latentspace.detach().cpu().numpy()
            )
            captured_data["reconstructions"][mod_name] = (
                model_output.reconstruction.detach().cpu().numpy()
            )
            if model_output.latent_mean is not None:
                captured_data["mus"][mod_name] = (
                    model_output.latent_mean.detach().cpu().numpy()
                )
            if model_output.latent_logvar is not None:
                captured_data["sigmas"][mod_name] = (
                    model_output.latent_logvar.detach().cpu().numpy()
                )

        return captured_data

    def _dynamics_to_result(self, split: str, epoch_dynamics: List[Dict]) -> None:
        """Aggregates and stores epoch dynamics into the Result object.

        Due to our multi modal Traning with unpaired data, we can see sample_ids more than once per epoch.
        For more consistent downstream analysis, we only keep the first occurence of this sample in the epoch
        and report the dynamics for this sample

        Args:
            split: The data split name (e.g., 'train', 'valid', 'test').
            epoch_dynamics: List of dictionaries capturing dynamics for each batch in the epoch.

        """
        final_data: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
        for batch_data in epoch_dynamics:
            for dynamic_type, mod_data in batch_data.items():
                for mod_name, data in mod_data.items():
                    final_data[dynamic_type][mod_name].append(data)

        sample_ids: Optional[Dict[str, np.ndarray]] = final_data.get("sample_ids")
        if sample_ids is None:
            raise ValueError("No Sample Ids in TrainingDynamics")
        concat_ids: Dict[str, np.ndarray] = {
            k: np.concatenate(v) for k, v in sample_ids.items()
        }
        unique_idx: Dict[str, np.ndarray] = {
            k: np.unique(v, return_index=True)[1] for k, v in concat_ids.items()
        }

        deduplicated_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        # all_dynamics_inner: Dict[str, Any] = {}
        for dynamic_type, dynamic_data in final_data.items():
            concat_dynamic: Dict[str, np.ndarray] = {
                k: np.concatenate(v) for k, v in dynamic_data.items()
            }
            # all_dynamics_inner[dynamic_type] = concat_dynamic
            deduplicated_helper: Dict[str, np.ndarray] = {
                k: v[unique_idx[k]] for k, v in concat_dynamic.items()
            }
            deduplicated_data[dynamic_type] = deduplicated_helper
            # Store the deduplicated data
        # self._all_dynamics[split][self._cur_epoch] = all_dynamics_inner
        self._result.latentspaces.add(
            epoch=self._cur_epoch,
            split=split,
            data=deduplicated_data.get("latentspaces", {}),
        )

        self._result.reconstructions.add(
            epoch=self._cur_epoch,
            split=split,
            data=deduplicated_data.get("reconstructions", {}),
        )

        if deduplicated_data.get("sample_ids"):
            self._result.sample_ids.add(
                epoch=self._cur_epoch,
                split=split,
                data=deduplicated_data["sample_ids"],
            )

        if deduplicated_data.get("mus"):
            self._result.mus.add(
                epoch=self._cur_epoch,
                split=split,
                data=deduplicated_data["mus"],
            )

        if deduplicated_data.get("sigmas"):
            self._result.sigmas.add(
                epoch=self._cur_epoch,
                split=split,
                data=deduplicated_data["sigmas"],
            )

    def _log_losses(
        self,
        split: str,
        total_loss: float,
        sub_losses: Dict[str, float],
        n_samples: int,
    ) -> None:
        """Logs and stores average losses for the epoch.
        Args:
            split: The data split name (e.g., 'train', 'valid').
            total_loss: The cumulative total loss for the epoch.
            sub_losses: Dictionary of accumulated sub-losses for the epoch.
            n_samples: Total number of samples processed in the epoch.

        """

        n_samples = max(n_samples, 1)
        print(f"split: {split}, n_samples: {n_samples}")
        # avg_total_loss = total_loss / n_samples ## Now already normalized
        self._result.losses.add(epoch=self._cur_epoch, split=split, data=total_loss)

        # avg_sub_losses = {
        #     k: v / n_samples if "_factor" not in k else v for k, v in sub_losses.items()
        # }
        self._result.sub_losses.add(
            epoch=self._cur_epoch,
            split=split,
            data=sub_losses,
        )
        self._fabric.print(
            f"Epoch {self._cur_epoch + 1}/{self._config.epochs} - {split.capitalize()} Loss: {total_loss:.4f}"
        )
        # Detailed sub-loss logging in one line
        sub_loss_str = ", ".join(
            [f"{k}: {v:.4f}" for k, v in sub_losses.items() if "_factor" not in k]
        )
        self._fabric.print(f"Sub-losses - {sub_loss_str}")

    def _store_checkpoint(self, split: str, epoch_dynamics: List[Dict]) -> None:
        """Stores model checkpoints and epoch dynamics into the Result object.

        Args:
            split: The data split name (e.g., 'train', 'valid').
            epoch_dynamics: List of dictionaries capturing dynamics for each batch in the epoch.

        """

        state_to_save = {
            mod_name: dynamics["model"].state_dict()
            for mod_name, dynamics in self._modality_dynamics.items()
        }
        state_to_save["latent_clf"] = self._latent_clf.state_dict()
        self._result.model_checkpoints.add(epoch=self._cur_epoch, data=state_to_save)
        self._dynamics_to_result(split=split, epoch_dynamics=epoch_dynamics)

    def _store_final_models(self) -> None:
        """Stores the final trained models into the Result object."""
        final_models = {
            mod_name: dynamics["model"]
            for mod_name, dynamics in self._modality_dynamics.items()
        }
        self._result.model = final_models

    def purge(self) -> None:
        """
        Cleans up all instantiated resources used during training, including
        all modality-specific models/optimizers and the adversarial classifier.
        """

        if hasattr(self, "_modality_dynamics"):
            for mod_name, dynamics in self._modality_dynamics.items():
                if "model" in dynamics and dynamics["model"] is not None:
                    # Remove the model and its optimizer from the dynamics dict
                    del dynamics["model"]
                if "optim" in dynamics and dynamics["optim"] is not None:
                    del dynamics["optim"]
            # Delete the container itself after cleaning contents
            del self._modality_dynamics

        if hasattr(self, "_latent_clf"):
            del self._latent_clf
        if hasattr(self, "_clf_optim"):
            del self._clf_optim

        if hasattr(self, "_trainloader"):
            del self._trainloader
        if hasattr(self, "_validloader") and self._validloader is not None:
            del self._validloader
        if hasattr(self, "_trainset"):
            del self._trainset
        if hasattr(self, "_validset"):
            del self._validset
        if hasattr(self, "_loss_fn"):
            del self._loss_fn
        if hasattr(self, "sub_loss"):  # Directly accessible sub_loss instance
            del self.sub_loss
        if hasattr(self, "_clf_loss_fn"):
            del self._clf_loss_fn

        if hasattr(self, "_all_dynamics"):
            del self._all_dynamics

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
