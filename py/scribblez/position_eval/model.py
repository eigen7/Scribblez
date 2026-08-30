"""Position evaluation model for Scrabble.

Architecture:
  - Spatial trunk: (planes, 15, 15) -> conv 3x3 -> 128 channels -> N residual blocks
  - Scalar injection: (scalars,) -> FC -> 128 -> broadcast-add to spatial features
  - Pooling: global average pool -> 128-d trunk vector
  - Six heads:
    * WLD (inference): FC -> 3 logits (win/draw/loss)
    * ScoreDiff (aux): FC -> 2 = [mean, std] of the final score differential.
      The mean is regressed against the observed differential and the std
      against the absolute residual of the mean, both with Huber loss in score
      points. The std stack reads a detached copy of the trunk summary, so its
      loss trains only that stack and never perturbs the trunk or the mean. The
      exported second value is the std directly.
    * Placement heads (aux): four categorical distributions over move
      FOOTPRINTS (training/footprint.h) -- a KataGo-policy-style Conv2d(C->13)
      giving per-(cell, orientation, k) logits, flattened to the anchored
      footprints, plus a pooled FC for the two catch-all classes (pass, and the
      win heads' not-win / the plays heads' dummy). The plays heads
      OppNextPlacement / SelfNextPlacement predict which footprint each player's
      next move is; the win heads OppWinPlacement / SelfWinPlacement predict
      Pr[footprint AND that player wins], carrying not-win on the extra class --
      the "opponent danger" / "self opportunity" signals of
      docs/sim_residual_feedback.md, now over move footprints instead of a
      per-cell marginal. Trained by masked softmax cross-entropy: the raw
      logits are the export, and each consumer masks illegal footprints before
      the softmax (the mask is not in the graph). The (15,15) per-cell marginal
      the dashboard shows is recovered downstream by summing footprint
      probability over the cells each footprint covers.

The two model input widths come from the engine session's input-encoding spec
(85 planes / 936 scalars, plus 27 scalars under the open-leaves arm) and, with
the six head output shapes, are fixed by the training pipeline and the C++
inference contract; the trunk between them is free to change.

docs/model_architectures.md diagrams this network; any change to the
architecture belongs in the same commit as the corresponding change there.
"""

import math

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


class FootprintPlacementHead(nn.Module):
    """One placement head: a categorical distribution over move footprints.

    Conv2d(C -> slots_per_cell) gives per-(cell, orientation, k) logits, whose
    (cell, slot) flattening is exactly training_targets.h's anchored-footprint
    class index (cell = row*side+col, then its slots). A pooled FC over the value
    summary emits the trailing catch-all logits (pass, not-win/dummy). The head
    is a raw-logit emitter; masking + softmax happen in the loss / consumers.
    """

    def __init__(self, trunk_channels: int, value_in: int, slots_per_cell: int, catch_all: int):
        super().__init__()
        self.conv = nn.Conv2d(trunk_channels, slots_per_cell, 1)
        self.catch_all_fc = nn.Linear(value_in, catch_all)

    def forward(self, x: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # (B, slots, H, W) -> (B, H, W, slots) -> (B, H*W*slots): cell-major,
        # slot-minor, matching footprint_class's (cell*slots + slot) layout.
        anchored = self.conv(x).permute(0, 2, 3, 1).reshape(b, -1)
        return torch.cat([anchored, self.catch_all_fc(value_in)], dim=1)  # (B, num_classes)


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

        # --- Heads ---
        # TODO: make the set of heads modular so experimenting with additional
        # auxiliary heads touches as few places as possible. Right now each head
        # is hardcoded in several spots that must stay in sync: its submodule here
        # in __init__, its output entry in forward(), its loss term and weight in
        # compute_loss(), and the class/forward docstrings. A registry of head
        # objects (each owning its modules, forward, and loss) iterated over in
        # these spots would let a new head be added in one place.
        # The value heads read a 3C summary: mean+max board pooling (2C) plus the
        # scalar projection (C), so the score-diff scalar reaches them directly.
        value_in = 3 * trunk_channels

        # WLD head (inference head): FC -> 3.
        self.wld_fc = nn.Sequential(
            nn.Linear(value_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Score-diff head (aux): two independent FC stacks over the value
        # summary. The mean stack regresses the final score differential; the
        # std stack regresses the absolute residual of the mean and reads a
        # detached copy of the value summary (see forward()), so the std loss
        # updates only this stack -- never the shared trunk or the mean.
        # forward() maps the std stack's output through softplus to a positive
        # std, so the exported second value is the std directly.
        self.sd_mean_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.sd_std_fc = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Placement heads (aux): one categorical footprint head per placement
        # name. Each owns its own Conv2d(C->slots) policy reduction (they predict
        # different distributions -- opp vs self, plays vs win -- so they do not
        # share a trunk reduction) plus a pooled FC for the catch-all logits.
        self.placement_heads = nn.ModuleDict(
            {
                name: FootprintPlacementHead(
                    trunk_channels, value_in, FOOTPRINT_SLOTS_PER_CELL, FOOTPRINT_CATCH_ALL
                )
                for name in PLACEMENT_HEAD_NAMES
            }
        )

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

        wld = self.wld_fc(value_in)

        # Score-diff head: [mean, std] in score points. The std stack reads a
        # detached copy of the value summary, so its loss never flows into the
        # trunk or the mean. softplus + floor keeps the std positive and away
        # from zero.
        sd_mean = self.sd_mean_fc(value_in)  # (B, 1)
        sd_std = F.softplus(self.sd_std_fc(value_in.detach())) + 1e-3  # (B, 1)
        sd = torch.cat([sd_mean, sd_std], dim=1)  # (B, 2): [mean, std]

        out = {"wld": wld, "score_diff": sd}
        for name in PLACEMENT_HEAD_NAMES:
            out[name] = self.placement_heads[name](x, value_in)  # (B, FOOTPRINT_CLASSES)
        return out


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


def compute_loss(
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
    """Compute combined loss for all heads.

    Args:
        outputs: model forward() result. "wld" (B,3 logits), "score_diff"
                 (B,2 = [mean, std]), and one (B, FOOTPRINT_CLASSES) raw
                 footprint-logit tensor per PLACEMENT_HEAD_NAMES entry.
        targets: dict with "wld" (B,3) one-hot, "score_diff" (B,1) the observed
                 final differential, the footprint class index per
                 PLACEMENT_HEAD_NAMES entry ("<name>", (B,1)), and the two side
                 legality masks ("opp_placement_mask" / "self_placement_mask",
                 (B, FOOTPRINT_CLASSES) in {0,1}) each head reads via its side.
        lambda_wld: weight applied to the WLD (value) loss. 1.0 in normal
                 training; drop it to isolate the other heads (a diagnostic that
                 leaves the value head untrained).
        lambda_next_placement: weight applied to each of the two plays-head
                 placement losses (opp and self).
        lambda_win_placement: weight applied to each of the two win-head
                 placement losses (opp and self).
        huber_delta_mean: Huber transition point (points) for the mean.
        huber_delta_std: Huber transition point (points) for the std.
        mask_placement: mask illegal footprints before the softmax (the
                 deployable default). False runs plain softmax-CE over all
                 classes -- the masked-vs-unmasked arm that keeps masking from
                 confounding the loss-geometry result.

    Returns:
        Dict with "total" plus one entry per head loss.
    """
    # WLD: cross-entropy against one-hot target.
    wld_target_idx = targets["wld"].argmax(dim=1)
    loss_wld = F.cross_entropy(outputs["wld"], wld_target_idx)

    # Score-diff: two Huber regressions in score points. The mean regresses the
    # observed differential. The std regresses the absolute residual of the mean
    # (scaled by MAD_TO_STD so its optimum is a Gaussian sigma). The std
    # prediction and the residual target are both detached from the trunk and
    # the mean, so the std loss trains only the std stack.
    sd_mean = outputs["score_diff"][:, 0]
    sd_std = outputs["score_diff"][:, 1]
    sd_target = targets["score_diff"].squeeze(1)
    loss_sd_mean = F.huber_loss(sd_mean, sd_target, delta=huber_delta_mean)
    std_target = (sd_mean.detach() - sd_target).abs() * MAD_TO_STD
    loss_sd_std = F.huber_loss(sd_std, std_target, delta=huber_delta_std)
    loss_sd = loss_sd_mean + loss_sd_std

    # Placement heads: masked softmax cross-entropy against the footprint class,
    # the plays heads weighted by lambda_next_placement and the win heads by
    # lambda_win_placement. Softmax's conserved mass is the point -- it replaces
    # the per-cell BCE's drifting, easy-negative-diluted geometry. Each head reads
    # its side's legality mask (win heads with the not-win class made legal); the
    # unmasked arm passes None.
    placement_losses = {
        name: _placement_ce(
            outputs[name],
            targets[name].squeeze(1).long(),
            _head_legal_mask(name, targets) if mask_placement else None,
        )
        for name in PLACEMENT_HEAD_NAMES
    }

    total = (
        lambda_wld * loss_wld
        + lambda_sd * loss_sd
        + lambda_next_placement
        * (placement_losses["opp_next_placement"] + placement_losses["self_next_placement"])
        + lambda_win_placement
        * (placement_losses["opp_win_placement"] + placement_losses["self_win_placement"])
    )

    return {
        "total": total,
        "wld": loss_wld,
        "score_diff": loss_sd,
        "score_diff_mean": loss_sd_mean,
        "score_diff_std": loss_sd_std,
        **placement_losses,
    }
