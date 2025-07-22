from collections import defaultdict
from typing import Optional, Tuple, Type, Union, Dict, Any

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
from autoencodix.trainers._general_trainer import GeneralTrainer
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

        self._init_buffers()
        trainsampler = CoverageEnsuringSampler(multimodal_dataset=trainset)
        collate_fn = create_multimodal_collate_fn(multimodal_dataset=trainset)
        self._trainloader = DataLoader(
            self._trainset, batch_sampler=trainsampler, collate_fn=collate_fn
        )
        # self._trainloader = DataLoader(
        #     self._validset,
        #     batch_sampler=self._validset.sampler,
        #     collate_fn=self._validset.collate_fn,
        # )

    def _init_modality_training(self):
        model_map = {
            DataSetTypes.NUM: VarixArchitecture,
            DataSetTypes.IMG: ImageVAEArchitecture,
        }
        self._modality_training_helper = {
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
            self._modality_training_helper[mod_name] = {
                "model": model,
                "optim": optimizer,
                "mp": [],
                "losses": [],
            }

    def _init_adversial_training(self):
        pass

    def _pre_epoch(self, batch: Dict[str, Dict[str, Any]]):
        for k, v in batch.items():
            model = self._modality_training_helper[k]["model"]
            data = v["data"]
            print(data.shape)

            mp = model(data)
            _, loss_stats = self.sub_loss(model_output=mp, targets=data)
            self._modality_training_helper[k]["losses"] = loss_stats
            self._modality_training_helper[k]["mp"] = mp

    def train(self):
        for batch in self._trainloader:
            self._pre_epoch(batch=batch)


            yield batch
    def _post_epoch(self):
