"""The frozen teacher's prior cache: per position, the teacher's trunk on the
root position (the reader's board tokens), and per candidate, the teacher's
predictions at its post-move position (the prior every arm starts from).

M1a uses one frozen teacher as the prior, the leaf model and the root encoder
(docs/plans/supreme_bot_m1a.md, Decisions). The cache is computed once per
.sprobe file and stored beside it as <stem>.sprior, an uncompressed .npz
written atomically.

A reader can also run without the teacher's predictions (prior "none"): every
candidate gets the same constant prior and the teacher only encodes the root
board (uninformative). The positive control of the M1a plan, "The
uniform-prior control", runs this way.

The teacher's torch checkpoint is its tag's rolling checkpoint, which holds
only the latest generation; loading checks that this is the generation the
corpus was pinned to, and the cache builder checks the torch model against
that generation's ONNX export, the leaf model the probes ran.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from scribblez.ffi import (
    decode_rows,
    encode_candidate_rows,
    get_input_shapes,
    input_floats,
    set_opp_leave_input,
)
from scribblez.paths import TagPaths
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES, PositionEvalModel
from scribblez.spatial_trunk import mean_max_pool, transformer_config
from scribblez.transfer_test.probes import ProbeFile

POSITION_EVAL = "position_eval"
PRIOR_EXT = ".sprior"

# PRIOR_NONE's constant predictions. An even game, with a little draw mass so
# the draw logit starts within reach of its rate (0.3% of m1a-endgame-train's
# label outcomes); the score difference's mean 0 and spread about a single
# endgame's (median 37 points there).
UNINFORMATIVE_WLD = (0.495, 0.01, 0.495)
UNINFORMATIVE_SCORE = (0.0, 37.0)

# The parity check's np.allclose tolerances. The relative term covers the
# score head, which reads in points; GPU convolutions run in TF32.
PARITY_ATOL = 2e-3
PARITY_RTOL = 1e-3


@dataclass
class Prior:
    """One .sprobe file's cache. Candidate rows follow the .sprobe's
    candidate order."""

    root_board: np.ndarray  # (P, 225, C) float16: trunk tokens, cell r*15+c
    root_summary: np.ndarray  # (P, C) float16: the trunk's scalar projection
    wld: np.ndarray  # (K, 3) float32: win/draw/loss probabilities
    score: np.ndarray  # (K, 2) float32: score-difference mean and std
    placement: np.ndarray  # (K, 4, classes) float16: footprint logits, PLACEMENT_HEAD_NAMES order


def uninformative(prior: Prior) -> Prior:
    """`prior` with the teacher's per-candidate predictions replaced by
    constants (UNINFORMATIVE_*, uniform footprints), the root tokens kept: a
    prior that says nothing about any candidate, read-only views."""
    k = len(prior.wld)
    return Prior(
        root_board=prior.root_board,
        root_summary=prior.root_summary,
        wld=np.broadcast_to(np.float32(UNINFORMATIVE_WLD), (k, 3)),
        score=np.broadcast_to(np.float32(UNINFORMATIVE_SCORE), (k, 2)),
        placement=np.broadcast_to(np.float16(0), prior.placement.shape),
    )


def prior_path(probes_path: Path) -> Path:
    return probes_path.with_suffix(PRIOR_EXT)


def load_teacher(teacher_tag: str, generation: int, mount_root: Path) -> PositionEvalModel:
    """The teacher's torch model, on the CPU in eval mode. Sets the FFI
    session's input arm to the teacher's, so it must precede any other engine
    call. Raises if the tag's checkpoint holds a different generation."""
    ckpt_path = TagPaths(teacher_tag, POSITION_EVAL, mount_root).rolling_checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["generation_index"] != generation + 1:
        raise ValueError(
            f"{ckpt_path} holds generation {ckpt['generation_index'] - 1}, not {generation}"
        )
    cfg = ckpt["config"]
    set_opp_leave_input(cfg["face_up_leaves"])
    shapes = {s.name: s.dims for s in get_input_shapes()}
    model = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=cfg["trunk_channels"],
        num_blocks=cfg["num_blocks"],
        use_film=cfg["use_film"],
        transformer=transformer_config(cfg),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def _split(rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Input rows (N, input_floats) -> the model's spatial and scalar inputs."""
    planes, side, _ = {s.name: s.dims for s in get_input_shapes()}["input_spatial"]
    n = planes * side * side
    spatial = torch.from_numpy(np.ascontiguousarray(rows[:, :n])).view(-1, planes, side, side)
    return spatial.to(device), torch.from_numpy(np.ascontiguousarray(rows[:, n:])).to(device)


@torch.no_grad()
def _forward(model: PositionEvalModel, rows: np.ndarray, device: torch.device):
    """(trunk map (N, C, 15, 15), scalar projection (N, C), head outputs)."""
    x, s = model.trunk(*_split(rows, device))
    return x, s, model._run_heads(x, torch.cat([mean_max_pool(x), s], dim=1))


def check_onnx_parity(model: PositionEvalModel, onnx_path: Path, rows: np.ndarray):
    """Raise unless the torch model's value outputs match the ONNX export's on
    `rows` (input rows, a few dozen suffice)."""
    device = next(model.parameters()).device
    _, _, outputs = _forward(model, rows, device)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spatial, scalar = _split(rows, torch.device("cpu"))
    want = dict(
        zip(
            [o.name for o in session.get_outputs()],
            session.run(None, {"input_spatial": spatial.numpy(), "input_scalar": scalar.numpy()}),
            strict=True,
        )
    )
    for name in ("wld", "score_diff"):
        got = outputs[name].cpu().numpy()
        if not np.allclose(got, want[name], atol=PARITY_ATOL, rtol=PARITY_RTOL):
            err = float(np.abs(got - want[name]).max())
            raise ValueError(f"teacher checkpoint differs from {onnx_path} on {name}: {err:.4g}")


def root_rows(probes: ProbeFile) -> np.ndarray:
    """The pre-move input rows of the file's positions."""
    rows = decode_rows(
        probes.path.with_suffix(".slog"),
        probes.positions["game_index"],
        probes.positions["turn_index"],
        post_move=False,
    )
    return rows[:, : input_floats()]


def candidate_rows(probes: ProbeFile) -> np.ndarray:
    """The post-move input rows of the file's candidates."""
    return encode_candidate_rows(
        probes.path.with_suffix(".slog"),
        probes.positions["game_index"],
        probes.positions["turn_index"],
        probes.positions["num_candidates"],
        probes.candidates["move"],
    )


def compute_prior(
    model: PositionEvalModel, probes: ProbeFile, device: torch.device, batch: int = 256
) -> Prior:
    roots = root_rows(probes)
    boards, summaries = [], []
    for i in range(0, len(roots), batch):
        x, s, _ = _forward(model, roots[i : i + batch], device)
        boards.append(x.permute(0, 2, 3, 1).flatten(1, 2).half().cpu().numpy())
        summaries.append(s.half().cpu().numpy())
    cands = candidate_rows(probes)
    wld, score, placement = [], [], []
    for i in range(0, len(cands), batch):
        _, _, out = _forward(model, cands[i : i + batch], device)
        wld.append(torch.softmax(out["wld"], dim=1).cpu().numpy())
        score.append(out["score_diff"].cpu().numpy())
        placement.append(
            torch.stack([out[h] for h in PLACEMENT_HEAD_NAMES], dim=1).half().cpu().numpy()
        )
    return Prior(
        root_board=np.concatenate(boards),
        root_summary=np.concatenate(summaries),
        wld=np.concatenate(wld),
        score=np.concatenate(score),
        placement=np.concatenate(placement),
    )


def write_prior(prior: Prior, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **vars(prior))
    os.replace(tmp, path)


def read_prior(path: Path) -> Prior:
    with np.load(path) as z:
        return Prior(**{k: z[k] for k in z.files})
