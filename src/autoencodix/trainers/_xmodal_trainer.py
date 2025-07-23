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
)
from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.modeling._classifier import Classifier
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._utils import flip_labels
from autoencodix.utils._losses import VarixLoss
from autoencodix.utils._model_output import ModelOutput
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


class XModalTrainer(GeneralTrainer):
    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[BaseDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        ontologies: Optional[tuple] = None,
        sub_loss_type: Type[BaseLoss] = VarixLoss,
    ):
        super().__init__(
            trainset, validset, result, config, model_type, loss_type, ontologies
        )

        self.latent_dim = config.latent_dim
        if ontologies is not None:
            if not hasattr(self._model, "latent_dim"):
                raise ValueError(
                    "Model must have a 'latent_dim' attribute when ontologies are provided."
                )
            self.latent_dim = self._model.latent_dim

        # we will set this later, in the predict method
        self.n_test: Optional[int] = None
        self.n_train = len(trainset.data) if trainset else 0
        self.n_valid = len(validset.data) if validset else 0
        self.n_features = trainset.get_input_dim() if trainset else 0
        self.device = next(self._model.parameters()).device
        self.sub_loss = sub_loss_type(config=self._config)
        self._clf_epoch_loss = torch.tensor(0)
        self._epoch_loss = torch.tensor(0) # main loss for XModalix

        self._init_buffers()
        trainsampler = CoverageEnsuringSampler(multimodal_dataset=trainset)  # type: ignore
        validsampler = CoverageEnsuringSampler(multimodal_dataset=validset)  # type: ignore
        collate_fn = create_multimodal_collate_fn(multimodal_dataset=trainset)  # type: ignore
        valid_collate_fn = create_multimodal_collate_fn(multimodal_dataset=validset)  # type: ignore
        self._trainloader = DataLoader(
            self._trainset, # type: ignore
            batch_sampler=trainsampler,
            collate_fn=collate_fn,
        )
        self._validloader = DataLoader(
            self._validset,  # type: ignore
            batch_sampler=validsampler,
            collate_fn=valid_collate_fn,
        )

    def _init_modality_training(self):
        model_map = {
            DataSetTypes.NUM: VarixArchitecture,
            DataSetTypes.IMG: ImageVAEArchitecture,
        }
        self._modality_dynamics = {
            mod_name: None for mod_name in self._trainset.datasets.keys()
        }
        for mod_name, ds in self._trainset.datasets.items():
            model_type = model_map.get(ds.mytype)
            if model_type is None:
                raise ValueError()
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

    def _init_adversial_training(self):
        self._latent_clf = Classifier(input_dim=self._config.latent_dim)
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
            print(data.shape)

            mp = model(data)
            loss, loss_stats = self.sub_loss(model_output=mp, targets=data)
            self._modality_dynamics[k]["loss_stats"] = loss_stats
            self._modality_dynamics[k]["loss"] = loss
            self._modality_dynamics[k]["mp"] = mp

    def _prep_adver_training(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate latent spaces and generate modality labels for all modalities.
        Args:
            None
        Returns:
            all_latents: Concatenated latent representations from all modalities.
            all_labels: Modality labels as a tensor of integers in [0, n_modalities).
        """
        all_latents: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        mod2idx = {
            mod_name: idx for idx, mod_name in enumerate(self._modality_dynamics.keys())
        }

        for mod_name, helper in self._modality_dynamics.items():
            output: ModelOutput = helper["mp"]
            latents = output.latentspace  # shape: [BatchSize, latent_dim]
            label_id = mod2idx[mod_name]

            all_latents.append(latents)
            all_labels.append(
                torch.full(
                    (latents.size(0),),
                    fill_value=label_id,
                    dtype=torch.long,
                    device=latents.device,
                )
            )

        return torch.cat(all_latents, dim=0), torch.cat(
            all_labels, dim=0
        )  # shape: [Total_batchsize]

    def _train_clf(self) -> torch.Tensor:
        clf_scores = self._latent_clf(self._latents)
        clf_loss = self._clf_loss_fn(clf_scores, self._labels)
        self._clf_epoch_loss += clf_loss
        self._clf_optim.zero_grad()
        clf_loss.backward()
        self._clf_optim.step()
        return clf_scores

    def _train_one_epoch(self):
        # reset losses for each new epoch
        self._clf_epoch_loss = torch.tensor(0)
        self._epoch_loss = torch.tensor(0)
        for batch in self._trainloader:
            self._modalities_forward(batch=batch)

            self._latents, self._labels = self._prep_adver_training()
            clf_scores = self._train_clf()
            batch_loss, loss_dict = self._loss_fn(
                batch=batch,
                modality_dynamics=self._modality_dynamics,
                clf_scores=clf_scores,
                labels=self._labels,
                clf_loss_fn=self._clf_loss_fn,
            )
        # TODO
        return batch_loss, loss_dict, batch

    def train(self):
        self._init_adversial_training()
        self._init_modality_training()
        # for epoch in range(self._config.epochs):
        # TODO 

        batch_loss, loss_dict, batch = self._train_one_epoch()
        return self._modality_dynamics, batch, loss_dict

    def _post_epoch(self):
        pass
