"""The max-move-per-lane model's lane-analysis eval set and prediction decoding.

The trainer runs the model over a fixed set of GCG positions at each checkpoint
and stores the decoded per-(position, lane) predictions; the dashboard's Lane
analysis view pairs them with the engine's ground truth from
`scribblez.ffi.analyze_gcg`. See docs/react_dashboard.md.
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
    """(names, inputs): the GCG file stems (the dashboard's position labels) and
    the positions' flat model inputs stacked as (N, F). Requires the lexicon."""
    names: list[str] = []
    rows: list[np.ndarray] = []
    for gcg in dataset_gcgs(dataset_dir):
        _bundle, inp = analyze_gcg(gcg.read_text())
        names.append(gcg.stem)
        rows.append(inp)
    return names, (np.stack(rows) if rows else np.zeros((0, 0), np.float32))


def split_input(inputs: np.ndarray, spatial_planes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split flat (N, F) inputs into (spatial (N, P, 15, 15), scalar (N, S)).
    The encoder lays out the spatial planes first, channel-major."""
    cells = BOARD_SIZE * BOARD_SIZE
    spatial = inputs[:, : spatial_planes * cells].reshape(
        -1, spatial_planes, BOARD_SIZE, BOARD_SIZE
    )
    scalar = inputs[:, spatial_planes * cells :]
    return spatial, scalar


@torch.no_grad()
def predict(model, inputs: np.ndarray, spatial_planes: int, device) -> dict:
    """Run `model` over the dataset and decode what the dashboard displays:

    occ        (N, 30, 15, 27) uint8   thresholded occupancy union (logit > 0)
    score_pmf  (N, 30, 100)    float32 softmax over the score bins
    has_move   (N, 30)         float32 sigmoid has-move probability
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
