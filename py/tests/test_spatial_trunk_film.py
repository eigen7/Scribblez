"""FiLM conditioning in the shared SpatialTrunk.

use_film upgrades the trunk's two scalar/global-context injection sites (the stem
injection and each global-pooling block) from additive (x + beta) to FiLM
((1 + gamma) * x + beta). The gamma projections are zero-initialised so a FiLM
trunk starts numerically identical to the additive one -- a strict superset. These
tests pin both halves of that contract: gamma is inert at init (so training starts
from the additive solution), and the additive path is exactly recovered when a FiLM
trunk carries additive weights with gamma at zero (so the beta/gamma split and its
ordering are wired correctly, not swapped).
"""

import torch
from scribblez.spatial_trunk import GlobalPoolingResBlock, SpatialTrunk

P, S, C, NB = 85, 163, 64, 10


def _trunk(use_film):
    torch.manual_seed(0)
    return SpatialTrunk(P, S, C, NB, use_film=use_film).eval()


def test_gamma_is_zero_initialised():
    """Every gamma source -- the stem gain and each global-pool block's gain half
    -- starts at exactly zero, so (1 + gamma) starts at 1."""
    trunk = _trunk(use_film=True)
    assert torch.count_nonzero(trunk.stem_gamma.weight) == 0
    assert torch.count_nonzero(trunk.stem_gamma.bias) == 0
    n_pool = 0
    for block in trunk.blocks:
        if isinstance(block, GlobalPoolingResBlock):
            sc = block.spatial_channels
            assert torch.count_nonzero(block.pool_fc.weight[sc:]) == 0
            assert torch.count_nonzero(block.pool_fc.bias[sc:]) == 0
            n_pool += 1
    assert n_pool == NB // 3  # every third block is a global-pooling block


def test_film_at_init_matches_additive_with_shared_weights():
    """Copy an additive trunk's weights into a FiLM trunk -- including only the
    beta half of each global-pool projection, leaving gamma at zero -- and the two
    produce identical features. Proves the multiplicative half is inert at gamma=0
    and that beta is read from the correct half (a swapped split would diverge)."""
    add = _trunk(use_film=False)
    film = _trunk(use_film=True)
    # Load every shared weight; the global-pool projections have a wider (2C) shape
    # under FiLM, so exclude them here and copy their beta half explicitly below.
    shared = {k: v for k, v in add.state_dict().items() if not k.endswith("pool_fc.weight")}
    shared = {k: v for k, v in shared.items() if not k.endswith("pool_fc.bias")}
    film.load_state_dict(shared, strict=False)
    for a_blk, f_blk in zip(add.blocks, film.blocks, strict=True):
        if isinstance(f_blk, GlobalPoolingResBlock):
            sc = f_blk.spatial_channels
            with torch.no_grad():
                f_blk.pool_fc.weight[:sc].copy_(a_blk.pool_fc.weight)
                f_blk.pool_fc.bias[:sc].copy_(a_blk.pool_fc.bias)

    torch.manual_seed(1)
    sp, sc = torch.randn(4, P, 15, 15), torch.randn(4, S)
    with torch.no_grad():
        (fa, sa), (ff, sf) = add(sp, sc), film(sp, sc)
    assert torch.allclose(fa, ff, atol=1e-6)
    assert torch.allclose(sa, sf, atol=1e-6)  # returned scalar projection (beta) unchanged


def _first_gpool_block(trunk):
    for block in trunk.blocks:
        if isinstance(block, GlobalPoolingResBlock):
            return block
    raise AssertionError("no global-pooling block in the trunk")


def test_nonzero_gamma_changes_the_output():
    """With gamma perturbed off zero the FiLM path actually modulates -- the
    capacity the zero init hides is reachable, not dead. Both injection sites are
    checked separately (a dropped multiply at either would otherwise be dead code
    that the zero-init tests above cannot catch): the stem gain, and a global-pool
    block's gain half."""
    sp, sc = torch.randn(2, P, 15, 15), torch.randn(2, S)

    film = _trunk(use_film=True)
    with torch.no_grad():
        base, _ = film(sp, sc)
        film.stem_gamma.weight.normal_(std=0.1)
        moved, _ = film(sp, sc)
    assert not torch.allclose(base, moved), "stem gamma is not applied"

    film = _trunk(use_film=True)
    block = _first_gpool_block(film)
    with torch.no_grad():
        base, _ = film(sp, sc)
        block.pool_fc.weight[block.spatial_channels :].normal_(std=0.1)  # the gamma half
        moved, _ = film(sp, sc)
    assert not torch.allclose(base, moved), "global-pool block gamma is not applied"
