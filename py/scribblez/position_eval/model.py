"""Position evaluation model for Scrabble.

Architecture:
  - Spatial trunk: (planes, 15, 15) -> conv 3x3 -> 128 channels -> N residual blocks
  - Scalar injection: (scalars,) -> FC -> 128 -> broadcast-add to spatial features
  - Pooling: global average pool -> 128-d trunk vector
  - Six output heads, each a Head subclass (below) owning its own forward and
    loss; the per-head mechanics live in those class docstrings. In brief, by
    role rather than mechanism:
    * WLD (inference): 3 win/draw/loss logits.
    * ScoreDiff (aux): [mean, std] of the final score differential; the std
      trains on a detached stack so the exported std never perturbs the mean.
    * Placement heads (aux): four categorical distributions over move FOOTPRINTS
      (training/footprint.h). The plays heads OppNextPlacement / SelfNextPlacement
      predict each player's next-move footprint; the win heads OppWinPlacement /
      SelfWinPlacement predict Pr[footprint AND that player wins], carrying
      not-win on the extra class -- the "opponent danger" / "self opportunity"
      signals of docs/sim_residual_feedback.md over move footprints. The (15,15)
      per-cell marginal the dashboard shows is recovered downstream by summing
      footprint probability over the cells each footprint covers.

The model holds the heads in a name-keyed registry (self.heads), and forward(),
compute_loss()'s total, and the loss/target key sets all iterate it -- so an
auxiliary head is added by adding one entry to _build_heads(). The placement
family is generated from the FFI-served PLACEMENT_HEAD_NAMES; WLD and score-diff
are singleton heads in the same registry.

The two model input widths come from the engine session's input-encoding spec
(85 planes / 936 scalars, plus 27 scalars under the open-leaves arm) and, with
the six head output shapes, are fixed by the training pipeline and the C++
inference contract; the trunk between them is free to change.

docs/model_architectures.md diagrams this network; any change to the
architecture belongs in the same commit as the corresponding change there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.ffi import format_layout
from scribblez.position_eval.supply_attention import TileSupplyAttention
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool

# For r ~ N(0, sigma), E|r| = sqrt(2/pi)*sigma. Regressing the std against the
# absolute residual would otherwise converge to ~0.8*sigma; this rescales the
# target so the optimum matches a Gaussian sigma (what consumers assume).
MAD_TO_STD = math.sqrt(math.pi / 2)  # ~1.2533


# The placement heads, in the order they appear in forward()'s output dict and
# the ONNX export. The names and order are training_targets.h's placement
# targets, served over the FFI -- the same single source the C++ TensorRT decode
# binds output tensors by, so the export and the engine cannot disagree.
PLACEMENT_HEAD_NAMES = tuple(format_layout()["constants"]["placement_head_names"])

# The footprint class space each placement head is a distribution over, read
# from the same FFI source as the C++ targets/outputs so widths cannot drift:
# num_classes = anchored footprints (side*side*slots_per_cell) + pass + extra.
_FOOTPRINT = format_layout()["constants"]["footprint"]
FOOTPRINT_CLASSES = _FOOTPRINT["num_classes"]
FOOTPRINT_SLOTS_PER_CELL = _FOOTPRINT["slots_per_cell"]
FOOTPRINT_ANCHORED = _FOOTPRINT["anchored"]
FOOTPRINT_EXTRA_CLASS = _FOOTPRINT["extra_class"]
# Catch-all classes past the anchored footprints: pass, then not-win/dummy.
FOOTPRINT_CATCH_ALL = FOOTPRINT_CLASSES - FOOTPRINT_ANCHORED

# The legality-mask targets: one per SIDE (opp / self), not per head, read from
# the same FFI source as the head names so the two cannot drift. A side's plays
# head and win head share the footprint legality and differ only at the extra
# (not-win) class, so training_targets.h emits one mask per side (in the
# plays-head form, extra illegal) and the loss makes extra legal for the win
# head. _head_mask_name/_head_is_win derive the head -> side mapping.
PLACEMENT_MASK_NAMES = tuple(format_layout()["constants"]["placement_mask_names"])


def _head_mask_name(head: str) -> str:
    return "opp_placement_mask" if head.startswith("opp") else "self_placement_mask"


def _head_is_win(head: str) -> bool:
    return "win" in head


@dataclass
class LossConfig:
    """Weights and Huber transition points for the combined post-move loss. Each
    head reads the field(s) it needs from this config in its loss()."""

    lambda_wld: float
    lambda_sd: float
    lambda_next_placement: float
    lambda_win_placement: float
    huber_delta_mean: float
    huber_delta_std: float
    mask_placement: bool

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(
            args.lambda_wld,
            args.lambda_sd,
            args.lambda_next_placement,
            args.lambda_win_placement,
            args.huber_delta_mean,
            args.huber_delta_std,
            args.mask_placement,
        )


class HeadLoss(NamedTuple):
    """One head's loss contribution: `weighted` is added into the optimized
    total; `reported` is its per-key (unweighted) losses to log."""

    weighted: torch.Tensor
    reported: dict[str, torch.Tensor]


# --- Heads ----------------------------------------------------------------


class Head(nn.Module):
    """Base class for a position-eval output head.

    A head's forward takes the trunk feature map `x` (B, C, 15, 15) and the value
    summary `value_in` (B, 3C) and returns its single output tensor (value heads
    ignore `x`), so the model can run every head with a uniform call. Subclasses
    set `name` (the output-dict key, also the ONNX output name), `loss_keys` (the
    per-key losses loss() reports -- (name,) except for a composite head), and
    `target_keys` (the target-dict keys loss() reads).
    """

    name: str
    loss_keys: tuple[str, ...]
    target_keys: tuple[str, ...]

    def loss(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], cfg: LossConfig
    ) -> HeadLoss:
        raise NotImplementedError


class WldHead(Head):
    """WLD (value) inference head: FC over the value summary -> 3 win/draw/loss
    logits, trained by softmax cross-entropy against the one-hot outcome.
    lambda_wld weights it in the total; drop it to 0 to isolate the other heads
    (a diagnostic that leaves the value head untrained)."""

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
    """Score-diff aux head: two independent FC stacks over the value summary
    producing [mean, std] of the final score differential.

    forward: the mean stack regresses the differential; the std stack reads a
    detached copy of the value summary and is mapped through softplus + a floor
    to a positive std, so the exported second value is the std directly and its
    loss never flows into the trunk or the mean.

    loss: two Huber regressions in score points -- the mean against the observed
    differential, the std against the absolute residual of the mean (scaled by
    MAD_TO_STD so its optimum is a Gaussian sigma), the std prediction and target
    both detached from trunk and mean. The reported score_diff is their sum;
    lambda_sd weights it in the total."""

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
    """One placement head: a categorical distribution over move footprints.

    forward: Conv2d(C -> slots_per_cell) gives per-(cell, orientation, k) logits,
    whose (cell, slot) flattening is exactly training_targets.h's
    anchored-footprint class index (cell = row*side+col, then its slots); a
    pooled FC over the value summary emits the trailing catch-all logits (pass,
    not-win/dummy). The head is a raw-logit emitter -- masking + softmax happen in
    the loss / consumers.

    loss: masked softmax cross-entropy against the footprint class. Softmax's
    conserved mass is the point -- it replaces the per-cell BCE's drifting,
    easy-negative-diluted geometry. The head's legality-mask side (opp/self), its
    not-win-class handling (win vs plays), and its loss weight
    (lambda_win_placement vs lambda_next_placement) all follow from its name."""

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
        # (B, slots, H, W) -> (B, H, W, slots) -> (B, H*W*slots): cell-major,
        # slot-minor, matching footprint_class's (cell*slots + slot) layout.
        anchored = self.conv(x).permute(0, 2, 3, 1).reshape(b, -1)
        return torch.cat([anchored, self.catch_all_fc(value_in)], dim=1)  # (B, num_classes)

    def loss(self, outputs, targets, cfg):
        legal = _head_legal_mask(self.name, targets) if cfg.mask_placement else None
        ce = _placement_ce(outputs[self.name], targets[self.name].squeeze(1).long(), legal)
        weight = cfg.lambda_win_placement if self._is_win else cfg.lambda_next_placement
        return HeadLoss(weight * ce, {self.name: ce})


def _build_heads(trunk_channels: int, value_in: int) -> list[Head]:
    """The head registry, in output (== ONNX) order: wld, score_diff, then the
    FFI-driven placement family. An auxiliary head is added by adding it here."""
    return [
        WldHead(value_in),
        ScoreDiffHead(value_in),
        *(PlacementHead(name, trunk_channels, value_in) for name in PLACEMENT_HEAD_NAMES),
    ]


class PositionEvalModel(nn.Module):
    """Position evaluation network with 6 heads."""

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int = 192,
        num_blocks: int = 10,
        board_size: int = 15,
        lexicon_module: nn.Module | None = None,
        use_film: bool = False,
        use_supply_attention: bool = False,
    ):
        super().__init__()
        self.board_size = board_size

        # Shared conv trunk: stem + scalar injection + residual tower. An optional
        # compiled-lexicon tool is fused per-cell inside the trunk (both orientations).
        # use_film makes the scalar/global-context injection multiplicative (FiLM).
        self.trunk = SpatialTrunk(
            spatial_planes,
            scalar_size,
            trunk_channels,
            num_blocks,
            lexicon_module=lexicon_module,
            use_film=use_film,
        )

        # Optional tile-supply cross-attention: refines the post-trunk feature map
        # by letting each square attend to per-letter availability tokens, so the
        # placement heads can gate a square's cross-check letters on whether those
        # tiles are actually available (see supply_attention.py). Zero-initialised,
        # so it is inert at init -- a strict superset of the no-attention model.
        self.supply_attention = (
            TileSupplyAttention(trunk_channels, scalar_size) if use_supply_attention else None
        )

        # Heads, keyed by output name (see _build_heads and the Heads section).
        # The value heads read a 3C summary -- mean+max board pooling (2C) plus
        # the scalar projection (C) -- so the score-diff scalar reaches them
        # directly.
        value_in = 3 * trunk_channels
        self.heads = nn.ModuleDict(
            {head.name: head for head in _build_heads(trunk_channels, value_in)}
        )

    def loss_keys(self) -> tuple[str, ...]:
        """The per-head loss keys accumulated each epoch ("total" is the
        optimized objective), derived from the heads so a new head extends it."""
        return ("total", *(key for head in self.heads.values() for key in head.loss_keys))

    def target_keys(self) -> tuple[str, ...]:
        """The batch target tensors the heads' losses consume, pulled from the
        batch dict by name (deduplicated across heads that share a side mask)."""
        return tuple(dict.fromkeys(key for head in self.heads.values() for key in head.target_keys))

    def _run_heads(self, x: torch.Tensor, value_in: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run every head over the trunk feature map `x` and value summary
        `value_in`, returning the output dict in head (== ONNX) order. Shared with
        subclasses that condition `x` / `value_in` before the heads."""
        return {name: head(x, value_in) for name, head in self.heads.items()}

    def forward(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_spatial: (B, spatial_planes, 15, 15)
            input_scalar:  (B, scalar_size)

        Returns:
            Dict with keys: "wld" (B,3 logits), "score_diff" (B,2 = [mean, std]),
            and one (B, FOOTPRINT_CLASSES) raw footprint-logit tensor per
            PLACEMENT_HEAD_NAMES entry.
        """
        # Shared conv trunk (s, the scalar projection, is reused by the value
        # heads below).
        x, s = self.trunk(input_spatial, input_scalar)

        # Optional tile-supply cross-attention refines the feature map before the
        # heads read it, so placement (and value) reflect letter availability.
        if self.supply_attention is not None:
            x = self.supply_attention(x, input_spatial, input_scalar)

        # Value summary: mean+max board pooling concatenated with the scalar
        # projection, so the heads see global board context and the raw scalars.
        value_in = torch.cat([mean_max_pool(x), s], dim=1)  # (B, 3C)
        return self._run_heads(x, value_in)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        lambda_wld: float = 1.0,
        lambda_sd: float = 1.0,
        lambda_next_placement: float = 0.5,
        lambda_win_placement: float = 0.5,
        huber_delta_mean: float = 10.0,
        huber_delta_std: float = 10.0,
        mask_placement: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Combined loss over the heads: the weighted sum each head contributes,
        plus every head's reported per-key losses.

        Each Head owns its own loss term and reads the weight it needs from the
        assembled LossConfig, so the total and the returned dict both derive from
        the head set -- there is no per-head term to hand-maintain here.

        Args:
            outputs: this model's forward() result. "wld" (B,3 logits),
                     "score_diff" (B,2 = [mean, std]), and one (B,
                     FOOTPRINT_CLASSES) raw footprint-logit tensor per
                     PLACEMENT_HEAD_NAMES entry.
            targets: dict with "wld" (B,3) one-hot, "score_diff" (B,1) the
                     observed final differential, the footprint class index per
                     PLACEMENT_HEAD_NAMES entry ("<name>", (B,1)), and the two
                     side legality masks ("opp_placement_mask" /
                     "self_placement_mask", (B, FOOTPRINT_CLASSES) in {0,1}) each
                     head reads via its side.
            lambda_wld: weight on the WLD (value) loss (WldHead).
            lambda_sd: weight on the score-diff loss (ScoreDiffHead).
            lambda_next_placement: weight on each plays-head placement loss (opp/self).
            lambda_win_placement: weight on each win-head placement loss (opp/self).
            huber_delta_mean: Huber transition point (points) for the score-diff mean.
            huber_delta_std: Huber transition point (points) for the score-diff std.
            mask_placement: mask illegal footprints before the softmax (the
                     deployable default). False runs plain softmax-CE over all
                     classes -- the masked-vs-unmasked arm that keeps masking
                     from confounding the loss-geometry result.

        Returns:
            Dict with "total" plus one entry per head-reported loss key.
        """
        cfg = LossConfig(
            lambda_wld,
            lambda_sd,
            lambda_next_placement,
            lambda_win_placement,
            huber_delta_mean,
            huber_delta_std,
            mask_placement,
        )
        total: torch.Tensor | None = None
        reported: dict[str, torch.Tensor] = {}
        for head in self.heads.values():
            head_loss = head.loss(outputs, targets, cfg)
            total = head_loss.weighted if total is None else total + head_loss.weighted
            reported.update(head_loss.reported)
        return {"total": total, **reported}


def _head_legal_mask(head: str, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """The (B, C) legality mask for one placement head: its side's mask, with the
    extra (not-win) class made legal for a win head. The stored side mask carries
    the plays-head form (extra illegal), so a plays head reads it unchanged."""
    side = targets[_head_mask_name(head)]
    if not _head_is_win(head):
        return side
    legal = side.clone()
    legal[:, FOOTPRINT_EXTRA_CLASS] = 1.0
    return legal


def _placement_ce(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    legal_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Masked softmax cross-entropy for one footprint head.

    logits (B, C) raw; target_idx (B,) the footprint class. When legal_mask is
    given (B, C in {0,1}), illegal footprints are driven to -inf before the
    softmax so they carry no probability or gradient, with the target class always
    kept first (the -log(0) NaN guard): the engine masks are sound
    over-approximations, but a data-dependent gap must degrade to an unmasked
    target, never to NaN. legal_mask None is the unmasked arm (plain softmax-CE).
    """
    if legal_mask is not None:
        legal_mask = legal_mask.clone()
        legal_mask.scatter_(1, target_idx.unsqueeze(1), 1.0)
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))
    return F.cross_entropy(logits, target_idx)
