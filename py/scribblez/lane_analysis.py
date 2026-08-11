"""Max-move-per-lane lane-analysis: the GCG position dataset and decoding a model's
predictions over it for the dashboard.

Ground truth (per-lane best moves, the true union/score) comes from the engine via
`scribblez.ffi.analyze_gcg`. This module owns the Python side: listing the dataset's
GCG positions, building the model-input batch from them, and decoding a model's
outputs into the per-(position, lane) predictions the dashboard stores. The trainer
writes those predictions at each checkpoint; the dashboard API reads them back and
pairs them with the engine ground truth. See docs/react_dashboard.md.
"""

from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

from scribblez.ffi import analyze_gcg
from scribblez.paths import REPO_ROOT

# The default frozen evaluation set: hand-built realistic endgame-ish positions.
DEFAULT_DATASET = REPO_ROOT / "positions" / "NWL23" / "max-move-per-lane-test-dataset"

BOARD_SIZE = 15
N_LANES = 30


def dataset_gcgs(dataset_dir: str | Path) -> list[Path]:
    """The dataset's GCG files in stable natural order (pos-1, pos-2, ..., pos-10)."""
    return natsorted(Path(dataset_dir).glob("*.gcg"), key=lambda p: p.name)


def load_inputs(dataset_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """(names, inputs): each position's flat model-input tensor, stacked (N, F).

    `names` are the GCG file stems (the dashboard's position labels). Requires the
    lexicon (analyze_gcg enumerates legal moves for the ground truth)."""
    names: list[str] = []
    rows: list[np.ndarray] = []
    for gcg in dataset_gcgs(dataset_dir):
        _bundle, inp = analyze_gcg(gcg.read_text())
        names.append(gcg.stem)
        rows.append(inp)
    return names, (np.stack(rows) if rows else np.zeros((0, 0), np.float32))


def split_input(inputs: np.ndarray, spatial_planes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split flat inputs (N, F) into the model's (spatial (N, P, 15, 15), scalar
    (N, S)) halves -- the encoder lays spatial planes (channel-major) before the
    scalar rack counts."""
    cells = BOARD_SIZE * BOARD_SIZE
    spatial = inputs[:, : spatial_planes * cells].reshape(
        -1, spatial_planes, BOARD_SIZE, BOARD_SIZE
    )
    scalar = inputs[:, spatial_planes * cells :]
    return spatial, scalar


@torch.no_grad()
def predict(model, inputs: np.ndarray, spatial_planes: int, device) -> dict:
    """Run `model` over the dataset inputs and decode per-(position, lane) outputs:

        occ        (N, 30, 15, 27) uint8   thresholded occupancy union (logit > 0)
        score_pmf  (N, 30, 100)    float32 softmax over the score bins
        has_move   (N, 30)         float32 sigmoid has-move probability

    These are exactly what the dashboard needs to show the predicted union (vs the
    true one), the score histogram, and the per-lane has-move call.
    """
    spatial, scalar = split_input(inputs, spatial_planes)
    sp = torch.from_numpy(np.ascontiguousarray(spatial)).to(device)
    sc = torch.from_numpy(np.ascontiguousarray(scalar)).to(device)
    out = model(sp, sc)
    return {
        "occ": (out["lane_occupancy_logits"] > 0).to(torch.uint8).cpu().numpy(),
        "score_pmf": torch.softmax(out["lane_score_logits"], dim=-1)
        .cpu()
        .numpy()
        .astype(np.float32),
        "has_move": torch.sigmoid(out["lane_has_move_logits"]).cpu().numpy().astype(np.float32),
    }
