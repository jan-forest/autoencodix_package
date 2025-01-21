import torch
from autoencodix.utils._model_output import ModelOutput
"""
One assert per test does not really make sense for this class (dataclass) as it is just a container for data.
"""

def test_model_output_initialization():
    batch_size, features = 32, 10
    reconstruction = torch.randn(batch_size, features)
    latentspace = torch.randn(batch_size, 5)  # smaller dimension for latent space

    # Test with minimum required arguments
    output = ModelOutput(reconstruction=reconstruction, latentspace=latentspace)
    assert isinstance(output, ModelOutput)
    assert torch.equal(output.reconstruction, reconstruction)
    assert torch.equal(output.latentspace, latentspace)
    assert output.z is None
    assert output.latent_mean is None
    assert output.latent_logvar is None
    assert output.additional_info is None


def test_model_output_with_all_parameters():
    # Create sample tensors
    batch_size, features = 32, 10
    reconstruction = torch.randn(batch_size, features)
    latentspace = torch.randn(batch_size, 5)
    z = torch.randn(batch_size, 5)
    latent_mean = torch.randn(batch_size, 5)
    latent_logvar = torch.randn(batch_size, 5)
    additional_info = {"loss": 0.5}

    # Test with all parameters
    output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=latentspace,
        z=z,
        latent_mean=latent_mean,
        latent_logvar=latent_logvar,
        additional_info=additional_info,
    )

    assert isinstance(output, ModelOutput)
    assert torch.equal(output.reconstruction, reconstruction)
    assert torch.equal(output.latentspace, latentspace)
    assert torch.equal(output.z, z)
    assert torch.equal(output.latent_mean, latent_mean)
    assert torch.equal(output.latent_logvar, latent_logvar)
    assert output.additional_info == additional_info
