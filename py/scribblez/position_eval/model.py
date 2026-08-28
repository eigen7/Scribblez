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
    * Placement masks (aux): one shared 1x1-conv reduction -> (4, 15, 15), the
      four channels split into named (15, 15) sigmoid masks. The marginals
      OppNextPlacement / SelfNextPlacement predict where each player's next
      move places tiles; the conjunctions OppWinPlacement / SelfWinPlacement
      predict Pr[player's next move occupies the cell AND that player wins] --
      the per-square "opponent danger" / "self opportunity" maps of
      docs/sim_residual_feedback.md. Pairing each conjunction with its marginal
      lets consumers separate "plays there often" from "wins when playing
      there".

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


# The placement-mask heads, in the channel order mask_conv emits them (also
# the order they appear in forward()'s output dict and the ONNX export). The
# names and order are training_targets.h's placement targets, served over the
# FFI -- the same single source the C++ TensorRT decode binds output tensors
# by, so the export and the engine cannot disagree.
MASK_HEAD_NAMES = tuple(format_layout()["constants"]["placement_head_names"])


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

        # Placement-mask heads (aux): one shared 1x1-conv reduction emitting a
        # (len(MASK_HEAD_NAMES), 15, 15) logit stack, split per head in
        # forward(). The masks read closely related board structure, so they
        # share the reduction rather than each owning one.
        self.mask_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, len(MASK_HEAD_NAMES), 1),
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
            and one (B,15,15) logit mask per MASK_HEAD_NAMES entry.
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
        masks = self.mask_conv(x)  # (B, len(MASK_HEAD_NAMES), 15, 15)

        # Score-diff head: [mean, std] in score points. The std stack reads a
        # detached copy of the value summary, so its loss never flows into the
        # trunk or the mean. softplus + floor keeps the std positive and away
        # from zero.
        sd_mean = self.sd_mean_fc(value_in)  # (B, 1)
        sd_std = F.softplus(self.sd_std_fc(value_in.detach())) + 1e-3  # (B, 1)
        sd = torch.cat([sd_mean, sd_std], dim=1)  # (B, 2): [mean, std]

        out = {"wld": wld, "score_diff": sd}
        for i, name in enumerate(MASK_HEAD_NAMES):
            out[name] = masks[:, i]  # (B, 15, 15)
        return out


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_wld: float = 1.0,
    lambda_sd: float = 1.0,
    lambda_next_placement: float = 0.5,
    lambda_win_placement: float = 0.5,
    huber_delta_mean: float = 10.0,
    huber_delta_std: float = 10.0,
    placement_pos_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute combined loss for all heads.

    Args:
        outputs: model forward() result. "wld" (B,3 logits), "score_diff"
                 (B,2 = [mean, std]), and one (B,15,15) logit mask per
                 MASK_HEAD_NAMES entry.
        targets: dict with "wld" (B,3) one-hot, "score_diff" (B,1) the observed
                 final differential, and a (B,15,15) binary mask per
                 MASK_HEAD_NAMES entry.
        lambda_wld: weight applied to the WLD (value) loss. 1.0 in normal
                 training; drop it to isolate the other heads (a diagnostic that
                 leaves the value head untrained).
        lambda_next_placement: weight applied to each of the two marginal
                 placement losses (opp and self).
        lambda_win_placement: weight applied to each of the two win-placement
                 conjunction losses (opp and self).
        huber_delta_mean: Huber transition point (points) for the mean.
        huber_delta_std: Huber transition point (points) for the std.
        placement_pos_weight: BCE pos_weight for the placement-mask heads. 1.0 is
                 the ordinary (calibrated) loss; >1 up-weights the sparse
                 target-1 cells so a rare high-value square is not drowned by the
                 ~98% empty cells -- at the cost of calibration, so it is a
                 diagnostic knob, not a deployable default.

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

    # Placement-mask heads: binary cross-entropy per cell, the marginals
    # weighted by lambda_next_placement and the conjunctions by
    # lambda_win_placement. placement_pos_weight optionally up-weights the
    # target-1 cells (a scalar tensor on the logits' device broadcasts over the
    # board); None leaves BCE calibrated.
    pos_weight = (
        torch.tensor(placement_pos_weight, device=outputs[MASK_HEAD_NAMES[0]].device)
        if placement_pos_weight != 1.0
        else None
    )
    mask_losses = {
        name: F.binary_cross_entropy_with_logits(
            outputs[name], targets[name], pos_weight=pos_weight
        )
        for name in MASK_HEAD_NAMES
    }

    total = (
        lambda_wld * loss_wld
        + lambda_sd * loss_sd
        + lambda_next_placement
        * (mask_losses["opp_next_placement"] + mask_losses["self_next_placement"])
        + lambda_win_placement
        * (mask_losses["opp_win_placement"] + mask_losses["self_win_placement"])
    )

    return {
        "total": total,
        "wld": loss_wld,
        "score_diff": loss_sd,
        "score_diff_mean": loss_sd_mean,
        "score_diff_std": loss_sd_std,
        **mask_losses,
    }
