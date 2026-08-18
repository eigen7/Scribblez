"""The evidence trainer's two checkpoint kinds, loaded into a MoveSetEvalModel:
the student's rolling checkpoint (a move_set_eval tag's checkpoints/model.pt
-- generation 0, the plain student with the fusion stage at its zero-init and
an untrained proves-best head) and the trainer's own per-pass checkpoints
(checkpoints/model_epoch_NNNN.pt: the full model's weights, with the student's
config under "student"). Both the trainer and the dashboard's trajectory pane
load through here, so they agree on what a checkpoint's config means."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scribblez.move_set_eval.model import MoveSetEvalModel

# What the model needs from the student's config to be rebuilt without it.
STUDENT_CONFIG_KEYS = (
    "spatial_planes",
    "scalar_size",
    "trunk_channels",
    "num_blocks",
    "num_heads",
    "contingent_features",
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
    )


def load_student(path: str, device, freeze: bool = True) -> tuple[MoveSetEvalModel, dict]:
    """The model initialized from a move_set_eval rolling checkpoint -- its
    backbone frozen at the student's weights unless `freeze` is False (the
    trainer's unfrozen mode) -- and that checkpoint's config (the arch and
    encoding arm the student was built against, which this model inherits)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_student(ckpt["model_state_dict"])
    if freeze:
        model.freeze_backbone()
    return model.to(device), cfg


@dataclass
class EvidenceCheckpoint:
    """A loaded checkpoint of either kind. `trained` says whether the fusion
    stage and proves-best head carry trained weights (a per-pass checkpoint)
    or sit at their initial values (the student itself: conditioning is the
    identity and the gain head is noise)."""

    model: MoveSetEvalModel
    student_cfg: dict
    trained: bool


def load_evidence_checkpoint(path: str, device) -> EvidenceCheckpoint:
    """Load either checkpoint kind (told apart by the per-pass config's
    "student" block) into an eval-mode frozen-backbone model."""
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
