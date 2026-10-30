"""normalize_scalar in the shared SpatialTrunk: the scalar projection's scale
cannot run away from the stem's, however far its weights grow."""

import torch
from scribblez.spatial_trunk import SpatialTrunk

P, S, C, NB = 6, 5, 16, 2


def _features_with_scaled_scalar_proj(normalize_scalar: bool, scale: float) -> torch.Tensor:
    """The trunk's features with the scalar projection's output layer scaled by
    `scale`, on fixed inputs and weights."""
    torch.manual_seed(0)
    trunk = SpatialTrunk(P, S, C, NB, normalize_scalar=normalize_scalar).eval()
    with torch.no_grad():
        trunk.scalar_proj[2].weight.mul_(scale)
        trunk.scalar_proj[2].bias.mul_(scale)
        torch.manual_seed(1)
        features, _ = trunk(torch.randn(3, P, 15, 15), torch.randn(3, S))
    return features


def test_a_grown_scalar_projection_leaves_the_normalized_trunk_unchanged():
    base = _features_with_scaled_scalar_proj(normalize_scalar=True, scale=1.0)
    grown = _features_with_scaled_scalar_proj(normalize_scalar=True, scale=1000.0)
    assert torch.allclose(base, grown, atol=1e-2)


def test_a_grown_scalar_projection_changes_the_unnormalized_trunk():
    base = _features_with_scaled_scalar_proj(normalize_scalar=False, scale=1.0)
    grown = _features_with_scaled_scalar_proj(normalize_scalar=False, scale=1000.0)
    assert not torch.allclose(base, grown, atol=1e-2)
