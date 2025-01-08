import torch.nn as nn
from typing import List

# internal check done
# write tests: TODO
class LayerFactory:
    """
    Factory for creating configurable neural network layers.
    """
    @staticmethod
    def get_layer_dimensions(
        feature_dim: int, 
        latent_dim: int, 
        n_layers: int, 
        enc_factor: float
    ) -> List[int]:
        """
        Calculate progressive layer dimensions.
        
        Parameters
        ----------
        feature_dim : int
            Input feature dimension
        latent_dim : int
            Target latent dimension
        n_layers : int
            Number of layers
        enc_factor : float
            Reduction factor for layer sizes
        
        Returns
        -------
        List[int]
            Calculated layer dimensions
        """
        layer_dimensions = [feature_dim]
        for _ in range(n_layers - 1):
            prev_layer_size = layer_dimensions[-1]
            next_layer_size = max(int(prev_layer_size / enc_factor), latent_dim)
            layer_dimensions.append(next_layer_size)
        layer_dimensions.append(latent_dim)
        return layer_dimensions
    
    @staticmethod
    def create_layer(
        in_features: int, 
        out_features: int, 
        dropout_p: float = 0.1,
        last_layer: bool = False
    ) -> List[nn.Module]:
        """
        Create a configurable layer with optional components.
        
        Parameters
        ----------
        in_features : int
            Input feature dimension
        out_features : int
            Output feature dimension
        dropout_p : float, optional
            Dropout probability, by default 0.1
        last_layer : bool, optional
            Flag to skip activation/dropout for final layer, by default False
        
        Returns
        -------
        List[nn.Module]
            List of layer components
        """
        if last_layer:
            return [nn.Linear(in_features, out_features)]
        
        return [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_p),
            nn.ReLU()
        ]