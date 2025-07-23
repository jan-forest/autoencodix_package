import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Binary classifier for adversarial training in cross-modal translation.

    Used to distinguish between latent representations from different modalities
    during adversarial training. The generator (autoencoders) tries to fool this
    classifier to align latent spaces.

    Attributes:
        input_dim: Dimension of input latent space.
        n_hidden: Number of hidden units in classifier layers.
    """

    def __init__(self, input_dim: int, n_hidden: int = 64) -> None:
        """Initialize latent space classifier.

        Args:
            input_dim: Dimension of input latent space.
            n_hidden: Number of hidden units.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_hidden = n_hidden

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=n_hidden, out_features=n_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=n_hidden // 2, out_features=1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of classifier.

        Args:
            x: Input latent representation.

        Returns:
            Classification probability (0=first modality, 1=second modality).
        """
        return self.classifier(x)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights using Xavier uniform initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
