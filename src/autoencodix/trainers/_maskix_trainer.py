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
        masking_fn: Optional[Any] = None,
        masking_fn_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
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
        self.masking_fn = masking_fn
        self.masking_fn_kwargs = masking_fn_kwargs
        if self.masking_fn is not None:
            self._validate_masking_fn()

    def _validate_masking_fn(self):
        """Test that user provided masking function accepts a torch.Tensor as input and returns a torch.Tensor."""
        import inspect

        sig = inspect.signature(self.masking_fn)
        if len(sig.parameters) < 1:
            raise ValueError(
                "The provided masking function must accept at least one argument (the input tensor)."
            )

        test_input = torch.randn(100, 30)
        test_output = self.masking_fn(test_input, **self.masking_fn_kwargs)
        if not isinstance(test_output, torch.Tensor):
            raise ValueError(
                "The provided masking function must return a torch.Tensor as output."
            )
        if test_output.shape != test_input.shape:
            raise ValueError(
                "The output shape of the masking function must match the input shape."
            )

    def maskix_hook(self, X: torch.Tensor) -> torch.Tensor:
        """Applies the Maskix corruption mechanism to the input data X.

        For each feature in X, with a probability defined in self._mask_probas,
        the feature value is replaced by a value from another randomly selected sample.
        This creates a corrupted version of the input data.

        Args:
            X: Input data tensor of shape (batch_size, num_features).
        Returns:
            A corrupted version of the input data tensor.
        """
        if self.masking_fn is None:
            return self._maskix_hook(X)
        return self.masking_fn(X, **self.masking_fn_kwargs)

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
