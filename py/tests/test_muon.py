"""The muon optimizer arm: MuonAdamW (generational/muon.py) and how
build_optimizer assigns a model's parameters to it (generational/optim.py).

The pieces that could silently go wrong are the orthogonalization (a wrong
update shape still trains, just badly), which parameters Muon owns (Muon on a
head or an embedding is a different optimizer), and the state dict (a resume
that dropped the momentum buffers would still run)."""

from dataclasses import dataclass

import pytest
import torch
from scribblez.generational.checkpoint import GenerationalState, resume, save
from scribblez.generational.muon import MuonAdamW, orthogonalize
from scribblez.generational.optim import WsdArm, build_optim_arm, build_optimizer, tower_matrices
from scribblez.generational.optimizer_arms import DEFAULT_LR, OPTIMIZER_MUON, OPTIMIZER_WSD
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.position_eval.model import PositionEvalModel
from scribblez.supply_registers import SCALAR_SIZE_OPEN_LEAVES
from scribblez.transformer_tower import TransformerConfig

_CPU = torch.device("cpu")
_PLANES = 87


@dataclass
class _Params:
    optimizer: str = OPTIMIZER_MUON
    lr: float = 0.0
    weight_decay: float = 0.01
    adam_beta2: float = 0.95
    batch_size: int = 4
    lr_warmup_rows: int = 0
    lr_cycle_rows: int = 800


def _transformer_model() -> PositionEvalModel:
    torch.manual_seed(0)
    return PositionEvalModel(
        _PLANES,
        SCALAR_SIZE_OPEN_LEAVES,
        trunk_channels=16,
        num_blocks=2,
        transformer=TransformerConfig(mid_channels=8, num_heads=2, ffn_channels=16),
    )


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(4, _PLANES, 15, 15, generator=g),
        torch.randn(4, SCALAR_SIZE_OPEN_LEAVES, generator=g),
    )


def _step(model, opt, seeds):
    for seed in seeds:
        opt.zero_grad()
        model(*_inputs(seed))["wld"].square().mean().backward()
        opt.step()


@pytest.mark.parametrize("shape", [(8, 24), (24, 8), (16, 16)])
def test_orthogonalize_is_near_semi_orthogonal(shape):
    """The reference coefficients put every singular value near 1 (roughly
    [0.7, 1.2]) in either orientation, across a 10x spread of input singular
    values. Five steps do not lift directions far below the largest (a 1000x
    spread leaves some near 0.1); Muon accepts that for speed."""
    gen = torch.Generator().manual_seed(0)
    k = min(shape)
    u, _ = torch.linalg.qr(torch.randn(shape[0], k, generator=gen))
    v, _ = torch.linalg.qr(torch.randn(shape[1], k, generator=gen))
    g = u @ torch.diag(torch.logspace(-1, 0, k)) @ v.T
    s = torch.linalg.svdvals(orthogonalize(g).float())
    assert s.min() > 0.6 and s.max() < 1.3


def test_orthogonalize_keeps_the_singular_vectors():
    """It changes the singular values only: U V^T of the input."""
    g = torch.randn(8, 24, generator=torch.Generator().manual_seed(0))
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    cos = torch.nn.functional.cosine_similarity(
        orthogonalize(g).float().flatten(), (u @ vh).flatten(), dim=0
    )
    assert cos > 0.97


def test_muon_owns_exactly_the_tower_matrices():
    model = _transformer_model()
    opt = build_optimizer(model, _Params())
    assert isinstance(opt, MuonAdamW)
    muon, adam_decay, adam_no_decay = opt.param_groups
    assert muon["muon"] and not adam_decay["muon"] and not adam_no_decay["muon"]
    assert {id(p) for p in muon["params"]} == {id(p) for p in tower_matrices(model)}
    names = {id(p): n for n, p in model.named_parameters()}
    owned = [names[id(p)] for p in muon["params"]]
    assert all(n.startswith("trunk.tower.blocks.") and n.endswith(".weight") for n in owned)
    assert any(n.endswith("q_proj.weight") for n in owned)
    assert any(n.endswith("ffn.down.weight") for n in owned)
    # Every parameter is in exactly one group.
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.01, 0.0]


def test_muon_needs_the_transformer_trunk():
    model = PositionEvalModel(_PLANES, SCALAR_SIZE_OPEN_LEAVES, trunk_channels=16, num_blocks=2)
    with pytest.raises(ValueError, match="transformer"):
        build_optimizer(model, _Params())


def test_muon_groups_reject_non_matrices():
    with pytest.raises(ValueError, match="2-D"):
        MuonAdamW([{"params": [torch.nn.Parameter(torch.zeros(3))], "muon": True}], lr=1e-3)


def test_muon_runs_on_the_wsd_schedule_at_the_wsd_rate():
    params = _Params()
    opt = build_optimizer(_transformer_model(), params)
    arm = build_optim_arm(None, params, opt, 0)
    assert isinstance(arm, WsdArm)
    assert DEFAULT_LR[OPTIMIZER_MUON] == DEFAULT_LR[OPTIMIZER_WSD]
    assert arm.current == DEFAULT_LR[OPTIMIZER_MUON]


def test_muon_update_has_adamw_scale():
    """The RMS-matching scale makes a Muon step move a matrix about as far as
    an AdamW step would (Adam's first step moves every entry by ~lr), so the
    two share one learning rate."""
    lr = 1e-2
    w = torch.nn.Parameter(torch.zeros(64, 32))
    w.grad = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    MuonAdamW([{"params": [w], "muon": True}], lr=lr).step()
    rms = w.detach().square().mean().sqrt().item()
    assert 0.1 * lr < rms < 0.5 * lr  # 0.2 * lr by construction, give or take the NS spread


def test_training_reduces_the_loss():
    model = _transformer_model()
    opt = build_optimizer(model, _Params(lr=3e-3))
    x = _inputs(0)
    with torch.no_grad():
        before = model(*x)["wld"].square().mean().item()
    _step(model, opt, [0] * 20)
    with torch.no_grad():
        assert model(*x)["wld"].square().mean().item() < 0.5 * before


def test_a_resumed_run_continues_identically(tmp_path):
    """Momentum buffers and Adam moments round-trip through the rolling
    checkpoint, which leaves a three-group state dict as it is."""
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    model = _transformer_model()
    opt = build_optimizer(model, _Params())
    _step(model, opt, [1, 2])
    save(paths, model, opt, GenerationalState(8, 1), {})
    _step(model, opt, [3, 4])

    resumed = _transformer_model()
    resumed_opt = build_optimizer(resumed, _Params())
    resume(paths, resumed, resumed_opt, _CPU)
    _step(resumed, resumed_opt, [3, 4])
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(a, b)


def test_a_stack_orthogonalizes_each_matrix_independently():
    """MuonAdamW orthogonalizes same-shaped matrices as one stack."""
    g = torch.randn(3, 8, 24, generator=torch.Generator().manual_seed(0))
    stacked = orthogonalize(g)
    for i in range(3):
        assert torch.allclose(stacked[i].float(), orthogonalize(g[i]).float(), atol=1e-2)
