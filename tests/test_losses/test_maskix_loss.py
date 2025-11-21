# tests/test_maskix_loss.py
import pytest
import torch
from autoencodix.utils._model_output import ModelOutput
from autoencodix.losses.maskix_loss import MaskixLoss  # adjust import as needed
from autoencodix.configs import DefaultConfig


@pytest.fixture
def default_config():
    config = DefaultConfig()
    config.delta_mask_predictor = 0.2
    config.delta_mask_corrupted = 0.75
    config.loss_reduction = "mean"
    config.reconstruction_loss = "mse"
    return config


@pytest.fixture
def maskix_loss(default_config):
    return MaskixLoss(config=default_config)


# ========================
# Valid Cases
# ========================
@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_total_loss_is_tensor(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    total_loss, _ = maskix_loss(model_output, targets, mask)
    assert isinstance(total_loss, torch.Tensor)


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_total_loss_is_scalar(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    total_loss, _ = maskix_loss(model_output, targets, mask)
    assert total_loss.dim() == 0


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_total_loss_is_float32(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    total_loss, _ = maskix_loss(model_output, targets, mask)
    assert total_loss.dtype == torch.float32


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_loss_dict_has_recon_loss(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    _, loss_dict = maskix_loss(model_output, targets, mask)
    assert "recon_loss" in loss_dict


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_loss_dict_has_mask_loss(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    _, loss_dict = maskix_loss(model_output, targets, mask)
    assert "mask_loss" in loss_dict


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_mask_loss_is_scalar(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    _, loss_dict = maskix_loss(model_output, targets, mask)
    assert loss_dict["mask_loss"].dim() == 0


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_recon_loss_is_scalar(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    _, loss_dict = maskix_loss(model_output, targets, mask)
    assert loss_dict["recon_loss"].dim() == 0


@pytest.mark.parametrize(
    "batch_size, feat_dim, mask_ratio",
    [(8, 100, 0.5), (1, 50, 0.3), (16, 2048, 0.7)],
)
def test_maskix_loss_valid_total_loss_positive(
    maskix_loss, batch_size, feat_dim, mask_ratio
):
    torch.manual_seed(42)
    # Create data
    targets = torch.randn(batch_size, feat_dim)
    reconstruction = torch.randn_like(targets)
    # Random binary mask (1 = corrupted/masked)
    mask = (torch.rand(batch_size, feat_dim) < mask_ratio).float()
    predicted_mask = torch.randn(batch_size, feat_dim)  # logits
    model_output = ModelOutput(
        reconstruction=reconstruction,
        latentspace=torch.randn(batch_size, 128),
        additional_info={"predicted_mask": predicted_mask},
    )
    total_loss, _ = maskix_loss(model_output, targets, mask)
    assert total_loss > 0


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_maskix_loss_different_reduction_is_scalar(default_config, reduction):
    default_config.loss_reduction = reduction
    loss_fn = MaskixLoss(config=default_config)
    B, D = 4, 10
    targets = torch.randn(B, D)
    recon = torch.randn(B, D)
    mask = torch.zeros(B, D)
    mask[:2] = 1.0  # first two rows fully masked
    pred_mask = torch.randn(B, D)
    model_output = ModelOutput(
        reconstruction=recon,
        latentspace=torch.randn(B, 64),
        additional_info={"predicted_mask": pred_mask},
    )
    loss, _ = loss_fn(model_output, targets, mask)
    assert loss.dim() == 0


# ========================
# Invalid Cases (should raise errors)
# ========================
def test_missing_additional_info(maskix_loss):
    model_output = ModelOutput(
        reconstruction=torch.randn(4, 10),
        latentspace=torch.randn(4, 64),
        additional_info=None,  # ← missing!
    )
    targets = torch.randn(4, 10)
    mask = torch.zeros(4, 10)
    with pytest.raises(ValueError, match="additional_info.*attribute"):
        maskix_loss(model_output, targets, mask)


def test_additional_info_not_dict(maskix_loss):
    model_output = ModelOutput(
        reconstruction=torch.randn(4, 10),
        latentspace=torch.randn(4, 64),
        additional_info="not_a_dict",  # ← wrong type
    )
    targets = torch.randn(4, 10)
    mask = torch.zeros(4, 10)
    with pytest.raises(TypeError, match="additional_info.*dict"):
        maskix_loss(model_output, targets, mask)


def test_missing_predicted_mask_in_additional_info(maskix_loss):
    model_output = ModelOutput(
        reconstruction=torch.randn(4, 10),
        latentspace=torch.randn(4, 64),
        additional_info={"wrong_key": torch.randn(4, 10)},  # ← no predicted_mask
    )
    targets = torch.randn(4, 10)
    mask = torch.zeros(4, 10)
    with pytest.raises(ValueError, match="predicted_mask.*additional_info"):
        maskix_loss(model_output, targets, mask)


def test_shape_mismatch_predicted_mask_vs_targets(maskix_loss):
    model_output = ModelOutput(
        reconstruction=torch.randn(8, 100),
        latentspace=torch.randn(8, 64),
        additional_info={"predicted_mask": torch.randn(8, 50)},  # ← wrong feature dim
    )
    targets = torch.randn(8, 100)
    mask = torch.zeros(8, 100)
    with pytest.raises(
        Exception  # torch will broadcast or mul will fail → catch runtime error
    ):
        # Actually, torch.mul will fail with size mismatch
        maskix_loss(model_output, targets, mask)


def test_non_binary_mask_behavior(maskix_loss):
    # Mask should be 0/1 (or float 0.0/1.0), but test with values outside
    model_output = ModelOutput(
        reconstruction=torch.randn(4, 10),
        latentspace=torch.randn(4, 64),
        additional_info={"predicted_mask": torch.randn(4, 10)},
    )
    targets = torch.randn(4, 10)
    mask = torch.full((4, 10), 0.5)  # ← not 0/1
    # Should still work (BCE handles it), but you could add a warning if desired
    loss, _ = maskix_loss(model_output, targets, mask)
    assert loss.dim() == 0  # no crash → good


@pytest.mark.parametrize("delta_corrupted", [0.0, 0.3, 0.5, 1.0, 1.5])
def test_extreme_delta_mask_corrupted_finite(default_config, delta_corrupted):
    default_config.delta_mask_corrupted = delta_corrupted
    loss_fn = MaskixLoss(config=default_config)
    B, D = 2, 5
    targets = torch.zeros(B, D)
    recon = torch.zeros(B, D)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0]])
    pred_mask = torch.full((B, D), 0.0)  # predicts no mask
    model_output = ModelOutput(
        reconstruction=recon,
        latentspace=torch.randn(B, 32),
        additional_info={"predicted_mask": pred_mask},
    )
    loss, _ = loss_fn(model_output, targets, mask)
    # Even with extreme weights, should not crash
    assert torch.isfinite(loss)


# Bonus: Test that weights are applied correctly (sanity check)
def test_weighting_logic_recon_loss_positive(maskix_loss):
    B, D = 1, 4
    targets = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    recon = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # big errors
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # first two corrupted
    pred_mask = torch.zeros_like(mask)  # predicts no corruption → high mask loss
    model_output = ModelOutput(
        reconstruction=recon,
        latentspace=torch.randn(1, 32),
        additional_info={"predicted_mask": pred_mask},
    )
    _, components = maskix_loss(model_output, targets, mask)
    # Since delta_mask_corrupted = 0.75 → corrupted positions get 0.75 weight, visible get 0.25
    # So recon error on visible should contribute less
    # This is a sanity check — actual loss includes mask loss too
    assert components["recon_loss"] > 0
