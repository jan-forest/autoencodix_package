from collections import defaultdict
from typing import Optional, Tuple, Type, Union, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset, DataSetTypes
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.data._multimodal_dataset import (
    CoverageEnsuringSampler,
    create_multimodal_collate_fn,
    MultiModalDataset,
)
from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.modeling._classifier import Classifier
from autoencodix.utils._losses import VarixLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


class XModalTrainer(BaseTrainer):
    def __init__(
        self,
        trainset: MultiModalDataset,
        validset: MultiModalDataset,
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        ontologies: Optional[Tuple] = None,
        sub_loss_type: Type[BaseLoss] = VarixLoss,
        model_map: Dict[DataSetTypes, Type[BaseAutoencoder]] = {
            DataSetTypes.NUM: VarixArchitecture,
            DataSetTypes.IMG: ImageVAEArchitecture,
        },
    ):
        self._trainset = trainset
        self._n_train_modalities = self._trainset.n_modalities  # type: ignore
        self.model_map = model_map
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
        self.sub_loss = sub_loss_type(config=self._config)
        self._clf_epoch_loss: float = 0.0
        self._epoch_loss: float = 0.0
        self._epoch_loss_valid: float = 0.0

    def _init_model_architecture(self, ontologies: Optional[Tuple] = None):
        """Override parent's model init - we don't use a single model"""
        pass

    def _setup_fabric(self):
        print("setup fabric")
        self._init_adversarial_training()
        self._init_modality_training()
        for dynamics in self._modality_dynamics.values():
            dynamics["model"], dynamics["optim"] = self._fabric.setup(
                dynamics["model"], dynamics["optim"]
            )
        self._latent_clf, self._clf_optim = self._fabric.setup(
            self._latent_clf, self._clf_optim
        )

        self._trainloader = self._fabric.setup_dataloaders(self._trainloader)
        if self._validloader is not None:
            self._validloader = self._fabric.setup_dataloaders(self._validloader)

        self._fabric.launch()

    def _init_loaders(self):
        print("called init loaders")
        trainsampler = CoverageEnsuringSampler(multimodal_dataset=self._trainset)
        validsampler = CoverageEnsuringSampler(multimodal_dataset=self._validset)
        collate_fn = create_multimodal_collate_fn(multimodal_dataset=self._trainset)
        valid_collate_fn = create_multimodal_collate_fn(
            multimodal_dataset=self._validset
        )
        self._trainloader = DataLoader(
            self._trainset,
            batch_sampler=trainsampler,
            collate_fn=collate_fn,
        )
        self._validloader = DataLoader(
            self._validset,
            batch_sampler=validsampler,
            collate_fn=valid_collate_fn,
        )

    def _init_modality_training(self):
        self._modality_dynamics = {
            mod_name: None for mod_name in self._trainset.datasets.keys()
        }
        for mod_name, ds in self._trainset.datasets.items():
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
            }

    def _init_adversarial_training(self):
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

    def _train_clf(self, latents: torch.Tensor, labels: torch.Tensor):
        """Performs a single optimization step for the classifier."""
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

    def _run_batch_forward(
        self, batch: Dict[str, Dict[str, Any]]
    ) -> Tuple[torch.Tensor, Dict]:
        """Performs a forward pass for validation."""
        self._modalities_forward(batch=batch)
        self._latents, self._labels = self._prep_adver_training()
        clf_scores = self._latent_clf(self._latents)
        batch_loss, loss_dict = self._loss_fn(
            batch=batch,
            modality_dynamics=self._modality_dynamics,
            clf_scores=clf_scores,
            labels=self._labels,
            clf_loss_fn=self._clf_loss_fn,
        )
        return batch_loss, loss_dict

    def _validate_one_epoch(self) -> Tuple[List[Dict], Dict[str, float], float, float]:
        """
        Runs a single, combined validation pass for the entire model,
        including dedicated metrics for the classifier.
        """
        # Set models to evaluation mode
        for dynamics in self._modality_dynamics.values():
            dynamics["model"].eval()
        self._latent_clf.eval()

        # Initialize accumulators
        self._epoch_loss_valid = 0.0
        total_clf_loss = 0.0

        epoch_dynamics: List[Dict] = []
        sub_losses: Dict[str, float] = defaultdict(float)

        with torch.no_grad():
            for batch in self._validloader:
                # --- 1. Perform Forward Pass (Same for all metrics) ---
                self._modalities_forward(batch=batch)
                latents, labels = self._prep_adver_training()
                clf_scores = self._latent_clf(latents)

                # --- 2. Calculate Main Combined Loss ---
                batch_loss, loss_dict = self._loss_fn(
                    batch=batch,
                    modality_dynamics=self._modality_dynamics,
                    clf_scores=clf_scores,
                    labels=labels,
                    clf_loss_fn=self._clf_loss_fn,
                )
                self._epoch_loss_valid += batch_loss.item()
                for k, v in loss_dict.items():
                    value = v.item() if hasattr(v, "item") else v
                    if "_factor" not in k:
                        sub_losses[k] += value
                    else:
                        sub_losses[k] = value

                # --- 3. Calculate Classifier-Only Loss and Accuracy ---
                clf_loss = self._clf_loss_fn(clf_scores, labels)
                total_clf_loss += clf_loss.item()

                _, predicted_labels = torch.max(clf_scores, 1)

                # --- 4. Capture Dynamics for Checkpointing (if needed) ---
                if self._is_checkpoint_epoch:
                    batch_capture = self._capture_dynamics(batch)
                    epoch_dynamics.append(batch_capture)

        sub_losses["clf_loss"] = total_clf_loss
        return epoch_dynamics, sub_losses

    # def _validate_one_epoch(self) -> Tuple[List[Dict], Dict[str, float]]:
    #     """Runs the validation loop for one epoch."""
    #     for dynamics in self._modality_dynamics.values():
    #         dynamics["model"].eval()
    #     self._latent_clf.eval()

    #     self._epoch_loss_valid = 0.0
    #     self._clf_epoch_loss_valid = 0.0
    #     epoch_dynamics: List[Dict] = []
    #     sub_losses: Dict[str, float] = defaultdict(float)

    #     with torch.no_grad():
    #         for batch in self._validloader:
    #             batch_loss, loss_dict = self._run_batch_forward(batch)
    #             self._epoch_loss_valid += batch_loss.item()
    #             for k, v in loss_dict.items():
    #                 value = v.item() if hasattr(v, "item") else v
    #                 if "_factor" not in k:
    #                     sub_losses[k] += value
    #                 else:
    #                     sub_losses[k] = (
    #                         value  # _factors are constants, so we can just overwrite this.
    #                     )
    #             if self._is_checkpoint_epoch:
    #                 batch_capture = self._capture_dynamics(batch)
    #                 epoch_dynamics.append(batch_capture)
    #         sub_losses["clf_loss"] = self._ep
    #     return epoch_dynamics, sub_losses

    def _train_one_epoch(self) -> Tuple[List[Dict], Dict[str, float]]:
        """Runs the training loop with corrected adversarial logic."""
        for dynamics in self._modality_dynamics.values():
            dynamics["model"].train()
        self._latent_clf.train()

        self._clf_epoch_loss = 0
        self._epoch_loss = 0
        epoch_dynamics: List[Dict] = []
        sub_losses: Dict[str, float] = defaultdict(float)

        for batch in self._trainloader:
            # --- Stage 1: forward for each data modality ---
            self._modalities_forward(batch=batch)

            # --- Stage 2: Train the Classifier ---
            latents, labels = self._prep_adver_training()
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
        sub_losses["clf_loss"] = self._clf_epoch_loss

        return epoch_dynamics, sub_losses

    def train(self):
        for epoch in range(self._config.epochs):
            self._cur_epoch = epoch
            self._is_checkpoint_epoch = self._should_checkpoint(epoch=epoch)
            self._fabric.print(f"--- Epoch {epoch + 1}/{self._config.epochs} ---")
            train_epoch_dynamics, train_sub_losses = self._train_one_epoch()

            self._log_losses(
                split="train",
                total_loss=self._epoch_loss,
                sub_losses=train_sub_losses,
                loader=self._trainloader,
            )

            if self._validset:
                valid_epoch_dynamics, valid_sub_losses = self._validate_one_epoch()
                self._log_losses(
                    split="valid",
                    total_loss=self._epoch_loss_valid,
                    sub_losses=valid_sub_losses,
                    loader=self._validloader,
                )
            if self._config.class_param:
                self._loss_fn.update_class_means(
                    epoch_dynamics=train_epoch_dynamics, device=self._fabric.device
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

        return self._result

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Multimodal decoding needs specific implementation")

    def predict(self, data: BaseDataset, model: torch.nn.Module) -> Result:
        raise NotImplementedError("Multimodal prediction needs specific implementation")

    def _capture_dynamics(
        self,
        batch_data: Union[Dict[str, Dict[str, Any]], Any],
    ) -> Union[Dict[str, Dict[str, np.ndarray]], Any]:
        captured_data: Dict[str, Dict[str, Any]] = {
            "latentspaces": {},
            "reconstructions": {},
            "mus": {},
            "sigmas": {},
            "sample_ids": {},
        }
        for mod_name, dynamics in self._modality_dynamics.items():
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
            sample_ids = batch_data[mod_name].get("sample_ids")
            if sample_ids is not None:
                captured_data["sample_ids"][mod_name] = np.array(sample_ids)
        return captured_data

    def _log_losses(
        self,
        split: str,
        total_loss: float,
        sub_losses: Dict[str, float],
        loader: DataLoader,
    ):
        # epoch loss is summed for each batch, so we average over n_batches
        n_batches = len(loader)  # len of PyTorch DataLoader gives n_batches

        avg_total_loss = total_loss / n_batches
        self._result.losses.add(epoch=self._cur_epoch, split=split, data=avg_total_loss)

        avg_sub_losses = {
            k: v / n_batches if "_factor" not in k else v for k, v in sub_losses.items()
        }
        self._result.sub_losses.add(
            epoch=self._cur_epoch,
            split=split,
            data=avg_sub_losses,
        )
        self._fabric.print(
            f"Epoch {self._cur_epoch + 1}/{self._config.epochs} - {split.capitalize()} Loss: {avg_total_loss:.4f}"
        )

    def _dynamics_to_result(self, split: str, epoch_dynamics: List[Dict]):
        final_data: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
        for batch_data in epoch_dynamics:
            for dynamic_type, mod_data in batch_data.items():
                for mod_name, data in mod_data.items():
                    final_data[dynamic_type][mod_name].append(data)

        self._result.latentspaces.add(
            epoch=self._cur_epoch,
            split=split,
            data={k: np.concatenate(v) for k, v in final_data["latentspaces"].items()},
        )
        self._result.reconstructions.add(
            epoch=self._cur_epoch,
            split=split,
            data={
                k: np.concatenate(v) for k, v in final_data["reconstructions"].items()
            },
        )
        if final_data.get("sample_ids"):
            self._result.sample_ids.add(
                epoch=self._cur_epoch,
                split=split,
                data={
                    k: np.concatenate(v) for k, v in final_data["sample_ids"].items()
                },
            )
        if final_data.get("mus"):
            self._result.mus.add(
                epoch=self._cur_epoch,
                split=split,
                data={k: np.concatenate(v) for k, v in final_data["mus"].items()},
            )
        if final_data.get("sigmas"):
            self._result.sigmas.add(
                epoch=self._cur_epoch,
                split=split,
                data={k: np.concatenate(v) for k, v in final_data["sigmas"].items()},
            )

    def _store_checkpoint(self, split: str, epoch_dynamics: List[Dict]):
        state_to_save = {
            mod_name: dynamics["model"].state_dict()
            for mod_name, dynamics in self._modality_dynamics.items()
        }
        state_to_save["latent_clf"] = self._latent_clf.state_dict()
        self._result.model_checkpoints.add(epoch=self._cur_epoch, data=state_to_save)
        self._dynamics_to_result(split=split, epoch_dynamics=epoch_dynamics)
