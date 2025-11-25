import torch
from typing import Optional, Type, Union, Tuple, Dict, Any, List
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig


class MaskixTrainer(GeneralTrainer):
    """Specialized trainer for Maskix (masked-based) autoencoders.

    TODO
    Attributes:
        Inherits all attributes from GeneralTrainer.
    """

    def __init__(
        self,
        trainset: Optional[BaseDataset],
        validset: Optional[BaseDataset],
        result: Result,
        config: DefaultConfig,
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        ontologies: Optional[Union[Tuple, List]] = None,
    ):
        """Initializes the OntixTrainer with the given datasets, model, and configuration.


        Args:
            trainset: The dataset used for training.
            validset: The dataset used for validation, if provided.
            result: An object to store and manage training results.
            config: Configuration object containing training hyperparameters and settings.
            model_type: The autoencoder model class to be trained.
            loss_type: The loss function class specific to the model.
        """
        super().__init__(
            trainset=trainset,
            validset=validset,
            result=result,
            config=config,
            model_type=model_type,
            loss_type=loss_type,
            ontologies=ontologies,
        )
        mask_probas_list: List[float] = [
            self._config.maskix_swap_prob
        ] * self._model.input_dim
        self._mask_probas = torch.tensor(mask_probas_list).to(self._model.device)

    def _maskix_hook_paper_code(self, X: torch.Tensor) -> torch.Tensor:
        """From the code of the publication, does not do column-wise shuffling, but the publication describes our _maskix_hook."""
        should_swap = torch.bernoulli(
            self._mask_probas.to(X.device) * torch.ones((X.shape)).to(X.device)
        )
        corrupted_X = torch.where(should_swap == 1, X[torch.randperm(X.shape[0])], X)
        return corrupted_X

    def _maskix_hook(
        self, X: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # expand probablities for bernoulli sampling to match input shape
        probs = self._mask_probas.expand(X.shape)

        # Create the Boolean Mask (1 = Swap, 0 = Keep)
        should_swap = torch.bernoulli(probs).bool()

        # COLUMN-WISE SHUFFLING
        # We generate a random float matrix and argsort it along dim=0.
        # This gives us independent random indices for every column.
        rand_indices = torch.rand(X.shape, device=X.device).argsort(dim=0)

        # Use gather to reorder X based on these random indices
        shuffled_X = torch.gather(X, 0, rand_indices)
        corrupted_X = torch.where(should_swap, shuffled_X, X)

        return corrupted_X
