import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Multi-class classifier for adversarial training in n-modal latent space alignment.


    Attributes:
        input_dim: Dimension of the input features.
        n_modalities: Number of modalities (classes) to classify.
        n_hidden: Number of hidden units in the classifier network.

    """

    def __init__(self, input_dim: int, n_modalities: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_modalities = n_modalities
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(n_hidden // 2, n_modalities),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            Output tensor of shape (batch_size, n_modalities) representing class scores.
        """
        return self.classifier(x)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
