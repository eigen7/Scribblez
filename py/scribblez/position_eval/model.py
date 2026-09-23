"""The position evaluation (value) model.

Given the board after a move, from the mover's POV, the model predicts the
game's outcome. It is the shared spatial trunk (spatial_trunk.py; the
transformer variant adds tile-supply register tokens, supply_registers.py)
followed by six heads:

  wld          (B, 3)            win/draw/loss logits; the value used at inference
  score_diff   (B, 2)            [mean, std] of the final score differential (aux)
  placement    (B, num_classes)  x4, raw logits over move footprints (aux):
      opp/self next   where that player's next move goes
      opp/self win    Pr[that footprint and that player wins]; the extra class
                      is "does not win"

Footprints are defined in engine/include/training/footprint.h. The dashboard's
per-cell placement maps are derived downstream by summing each footprint's
probability over the cells it covers.

The heads live in a name-keyed registry (self.heads) that forward, the loss and
the loss/target key sets all iterate, so adding a head means adding an entry to
_build_heads(). Placement head names come from the engine via the FFI.

The input widths (87 planes, 136 scalars, or 163 under the open-leaves arm) and
the head outputs are fixed by the engine's encoder and the C++ inference
contract; the trunk between them is free to change.

docs/model_architectures.md diagrams this network; keep it in sync.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.ffi import format_layout
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool
from scribblez.supply_registers import TileSupplyRegisters
from scribblez.transformer_tower import TransformerConfig

# For r ~ N(0, sigma), E|r| = sqrt(2/pi) * sigma, so regressing the std head on
# |residual| would converge to ~0.8 sigma. Scaling the target by this factor
# makes the optimum a Gaussian sigma, which is what consumers assume.
MAD_TO_STD = math.sqrt(math.pi / 2)  # ~1.2533


# The placement heads, in forward()'s output order and the ONNX export's. The
# engine serves them (training_targets.h) and binds its TensorRT outputs by the
# same names, so the export and the engine cannot disagree.
PLACEMENT_HEAD_NAMES = tuple(format_layout()["constants"]["placement_head_names"])

# The footprint class space, from the engine: anchored footprints
# (side * side * slots_per_cell), then pass, then extra (not-win).
_FOOTPRINT = format_layout()["constants"]["footprint"]
FOOTPRINT_CLASSES = _FOOTPRINT["num_classes"]
FOOTPRINT_SLOTS_PER_CELL = _FOOTPRINT["slots_per_cell"]
FOOTPRINT_ANCHORED = _FOOTPRINT["anchored"]
FOOTPRINT_EXTRA_CLASS = _FOOTPRINT["extra_class"]
FOOTPRINT_CATCH_ALL = FOOTPRINT_CLASSES - FOOTPRINT_ANCHORED

# The legality-mask targets, one per side (opp / self) rather than per head. A
# side's plays and win heads share footprint legality and differ only at the
# extra (not-win) class, so the engine emits one mask per side with extra
# illegal, and _head_legal_mask makes extra legal for the win head.
PLACEMENT_MASK_NAMES = tuple(format_layout()["constants"]["placement_mask_names"])


def _head_mask_name(head: str) -> str:
    return "opp_placement_mask" if head.startswith("opp") else "self_placement_mask"


def _head_is_win(head: str) -> bool:
    return "win" in head


@dataclass
class LossConfig:
    """Loss weights and Huber transition points (in score points); each head
    reads the fields it needs."""

    lambda_wld: float
    lambda_sd: float
    lambda_next_placement: float
    lambda_win_placement: float
    huber_delta_mean: float
    huber_delta_std: float

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(
            args.lambda_wld,
            args.lambda_sd,
            args.lambda_next_placement,
            args.lambda_win_placement,
            args.huber_delta_mean,
            args.huber_delta_std,
        )


class HeadLoss(NamedTuple):
    """One head's loss: `weighted` goes into the optimized total; `reported`
    holds its unweighted per-key losses for logging."""

    weighted: torch.Tensor
    reported: dict[str, torch.Tensor]


# --- Heads ----------------------------------------------------------------


class Head(nn.Module):
    """Base class for a position-eval output head.

    forward(x, value_in) takes the trunk feature map (B, C, 15, 15) and the value
    summary (B, 3C) and returns the head's one output tensor. Subclasses set
    `name` (the output-dict key and ONNX output name), `loss_keys` (the keys
    loss() reports), and `target_keys` (the target-dict keys loss() reads).
    """

    name: str
    loss_keys: tuple[str, ...]
    target_keys: tuple[str, ...]

    def loss(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], cfg: LossConfig
    ) -> HeadLoss:
        raise NotImplementedError


class WldHead(Head):
    """Win/draw/loss logits, trained by cross-entropy against the outcome."""

    name = "wld"
    loss_keys = ("wld",)
    target_keys = ("wld",)

    def __init__(self, value_in: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(value_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        return self.fc(value_in)

    def loss(self, outputs, targets, cfg):
        ce = F.cross_entropy(outputs["wld"], targets["wld"].argmax(dim=1))
        return HeadLoss(cfg.lambda_wld * ce, {"wld": ce})


class ScoreDiffHead(Head):
    """[mean, std] of the final score differential, from two separate MLPs.

    Both are Huber regressions: the mean against the observed differential, the
    std against the mean's absolute residual scaled by MAD_TO_STD. The std's
    input and target are detached, so its loss trains only the std MLP and never
    the trunk or the mean."""

    name = "score_diff"
    loss_keys = ("score_diff", "score_diff_mean", "score_diff_std")
    target_keys = ("score_diff",)

    def __init__(self, value_in: int):
        super().__init__()
        self.mean_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.std_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        mean = self.mean_fc(value_in)  # (B, 1)
        std = F.softplus(self.std_fc(value_in.detach())) + 1e-3  # (B, 1)
        return torch.cat([mean, std], dim=1)  # (B, 2): [mean, std]

    def loss(self, outputs, targets, cfg):
        mean = outputs["score_diff"][:, 0]
        std = outputs["score_diff"][:, 1]
        target = targets["score_diff"].squeeze(1)
        loss_mean = F.huber_loss(mean, target, delta=cfg.huber_delta_mean)
        std_target = (mean.detach() - target).abs() * MAD_TO_STD
        loss_std = F.huber_loss(std, std_target, delta=cfg.huber_delta_std)
        total = loss_mean + loss_std
        reported = {"score_diff": total, "score_diff_mean": loss_mean, "score_diff_std": loss_std}
        return HeadLoss(cfg.lambda_sd * total, reported)


class PlacementHead(Head):
    """Raw logits over move footprints for one placement head.

    A 1x1 conv emits the anchored classes, one channel per slot at each cell;
    a linear layer over the value summary emits the two catch-all classes.
    Masking and softmax happen in the loss and in downstream consumers.

    Trained by cross-entropy over legal footprints. The head's name determines
    its side's legality mask, whether the not-win class is legal, and its loss
    weight."""

    def __init__(self, name: str, trunk_channels: int, value_in: int):
        super().__init__()
        self.name = name
        self.loss_keys = (name,)
        self.target_keys = (name, _head_mask_name(name))
        self._is_win = _head_is_win(name)
        self.conv = nn.Conv2d(trunk_channels, FOOTPRINT_SLOTS_PER_CELL, 1)
        self.catch_all_fc = nn.Linear(value_in, FOOTPRINT_CATCH_ALL)

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # Cell-major, slot-minor: the engine's (cell * slots + slot) class index.
        anchored = self.conv(x).permute(0, 2, 3, 1).reshape(b, -1)
        return torch.cat([anchored, self.catch_all_fc(value_in)], dim=1)  # (B, num_classes)

    def loss(self, outputs, targets, cfg):
        legal = _head_legal_mask(self.name, targets)
        ce = _placement_ce(outputs[self.name], targets[self.name].squeeze(1).long(), legal)
        weight = cfg.lambda_win_placement if self._is_win else cfg.lambda_next_placement
        return HeadLoss(weight * ce, {self.name: ce})


def _build_heads(trunk_channels: int, value_in: int) -> list[Head]:
    """The heads, in output (and ONNX) order."""
    return [
        WldHead(value_in),
        ScoreDiffHead(value_in),
        *(PlacementHead(name, trunk_channels, value_in) for name in PLACEMENT_HEAD_NAMES),
    ]


class PositionEvalModel(nn.Module):
    """Position evaluation network; see the module docstring."""

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int = 192,
        num_blocks: int = 10,
        board_size: int = 15,
        lexicon_module: nn.Module | None = None,
        use_film: bool = False,
        transformer: TransformerConfig | None = None,
    ):
        super().__init__()
        self.board_size = board_size

        # The transformer tower gets the tile-supply register tokens.
        self.trunk = SpatialTrunk(
            spatial_planes,
            scalar_size,
            trunk_channels,
            num_blocks,
            lexicon_module=lexicon_module,
            use_film=use_film,
            transformer=transformer,
            registers=(TileSupplyRegisters(trunk_channels, scalar_size) if transformer else None),
            board_size=board_size,
        )

        # The value summary (see forward) is mean+max board pooling (2C) plus
        # the scalar projection (C), so scalars such as the score differential
        # reach the value heads directly.
        value_in = 3 * trunk_channels
        self.heads = nn.ModuleDict(
            {head.name: head for head in _build_heads(trunk_channels, value_in)}
        )

    def loss_keys(self) -> tuple[str, ...]:
        """ "total" (the optimized objective) plus every head's reported loss keys."""
        return ("total", *(key for head in self.heads.values() for key in head.loss_keys))

    def target_keys(self) -> tuple[str, ...]:
        """The batch keys the heads' losses read, deduplicated."""
        return tuple(dict.fromkeys(key for head in self.heads.values() for key in head.target_keys))

    def _run_heads(self, x: torch.Tensor, value_in: torch.Tensor) -> dict[str, torch.Tensor]:
        """The output dict, in head order. A separate method so subclasses that
        condition `x` / `value_in` can reuse it."""
        return {name: head(x, value_in) for name, head in self.heads.items()}

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """(B, spatial_planes, 15, 15), (B, scalar_size) -> the head outputs,
        keyed by head name (shapes in the module docstring)."""
        x, s = self.trunk(input_spatial, input_scalar)
        value_in = torch.cat([mean_max_pool(x), s], dim=1)  # (B, 3C)
        return self._run_heads(x, value_in)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        cfg: LossConfig,
    ) -> dict[str, torch.Tensor]:
        """The weighted total ("total") plus every head's reported losses.

        targets:
            wld                   (B, 3)            one-hot outcome
            score_diff            (B, 1)            final score differential
            <placement head>      (B, 1)            footprint class index
            opp_placement_mask,
            self_placement_mask   (B, num_classes)  1 = legal footprint
        """
        total: torch.Tensor | None = None
        reported: dict[str, torch.Tensor] = {}
        for head in self.heads.values():
            head_loss = head.loss(outputs, targets, cfg)
            total = head_loss.weighted if total is None else total + head_loss.weighted
            reported.update(head_loss.reported)
        return {"total": total, **reported}


def _head_legal_mask(head: str, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, num_classes) legality mask for one placement head: its side's mask,
    with the not-win class made legal for a win head."""
    side = targets[_head_mask_name(head)]
    if not _head_is_win(head):
        return side
    legal = side.clone()
    legal[:, FOOTPRINT_EXTRA_CLASS] = 1.0
    return legal


def _placement_ce(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over the legal footprints only: illegal logits are set to
    -inf, so they get no probability and no gradient.

    The target class is always kept legal. The engine's masks should
    over-approximate legality, but if one ever excludes the played footprint,
    the loss must stay finite rather than become -log(0).
    """
    legal_mask = legal_mask.clone()
    legal_mask.scatter_(1, target_idx.unsqueeze(1), 1.0)
    logits = logits.masked_fill(legal_mask == 0, float("-inf"))
    return F.cross_entropy(logits, target_idx)
