import torch
import numpy as np
from typing import Optional, Type, Union, Tuple
from collections import defaultdict
from torch.utils.data import DataLoader

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._model_output import ModelOutput
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.data._multimodal_dataset import CoverageEnsuringSampler, create_multimodal_collate_fn


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

        self._init_buffers()
        trainsampler = CoverageEnsuringSampler(multimodal_dataset=trainset)
        collate_fn = create_multimodal_collate_fn(multimodal_dataset=trainset)
        self._trainloader = DataLoader(
            self._trainset,
            batch_sampler=trainsampler,
            collate_fn=collate_fn
        )
        # self._trainloader = DataLoader(
        #     self._validset,
        #     batch_sampler=self._validset.sampler,
        #     collate_fn=self._validset.collate_fn,
        # )

    def train(self):
        for batch in self._trainloader:
            # return batch
            yield batch
