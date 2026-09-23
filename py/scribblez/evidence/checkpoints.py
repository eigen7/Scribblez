"""Loads the evidence trainer's two checkpoint kinds into a MoveSetEvalModel:

  * the student's rolling checkpoint (a move_set_eval tag's
    checkpoints/model.pt): the plain student, with the fusion stage at its
    zero-init and an untrained proves-best head;
  * the evidence trainer's per-pass checkpoints
    (checkpoints/model_epoch_NNNN.pt): the full model, with the student's
    config under "student".

The trainer and the dashboard's trajectory pane both load through here, so
they read a checkpoint's config the same way."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.spatial_trunk import transformer_config

# The student-config keys needed to rebuild the model without the student's
# checkpoint.
STUDENT_CONFIG_KEYS = (
    "spatial_planes",
    "scalar_size",
    "trunk_channels",
    "num_blocks",
    "num_heads",
    "trunk",
    "transformer_mid_channels",
    "transformer_heads",
    "transformer_ffn_channels",
    "open_leaves",
    "move_encoding_version",
)


def build_model(student_cfg: dict) -> MoveSetEvalModel:
    return MoveSetEvalModel(
        spatial_planes=student_cfg["spatial_planes"],
        scalar_size=student_cfg["scalar_size"],
        trunk_channels=student_cfg["trunk_channels"],
        num_blocks=student_cfg["num_blocks"],
        num_heads=student_cfg["num_heads"],
        transformer=transformer_config(student_cfg),
    )


def load_student(path: str, device, freeze: bool = True) -> tuple[MoveSetEvalModel, dict]:
    """(model initialized from a move_set_eval rolling checkpoint, that
    checkpoint's config). The backbone is frozen unless `freeze` is False.
    The model inherits the student's architecture and input encoding from the
    config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_student(ckpt["model_state_dict"])
    if freeze:
        model.freeze_backbone()
    return model.to(device), cfg


@dataclass
class EvidenceCheckpoint:
    """A loaded checkpoint of either kind. `trained` is False for the student
    itself, whose conditioning is the identity and whose gain head is
    untrained noise."""

    model: MoveSetEvalModel
    student_cfg: dict
    trained: bool


def load_evidence_checkpoint(path: str, device) -> EvidenceCheckpoint:
    """Load either checkpoint kind (per-pass configs have a "student" block)
    into an eval-mode, frozen-backbone model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if "student" in cfg:
        student_cfg = cfg["student"]
        model = build_model(student_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.freeze_backbone()
        return EvidenceCheckpoint(model.to(device).eval(), student_cfg, trained=True)
    model, student_cfg = load_student(path, device)
    return EvidenceCheckpoint(model.eval(), student_cfg, trained=False)
