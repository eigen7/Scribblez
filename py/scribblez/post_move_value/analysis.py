"""Post-move-value analysis: the GCG position dataset and decoding a model's
predictions over it for the dashboard.

Ground truth (per-position win/loss/draw + the exact final-score-delta distribution)
is precomputed offline by the `monte_carlo_sim_tool` and committed alongside the
dataset as `monte-carlo-sim-results.json`. This module owns the Python side: listing
the dataset's GCG positions, building the model-input batch from them (via
`scribblez.ffi.analyze_post_move_gcg`), and decoding a model's outputs into the
per-position predictions the dashboard stores. The trainer writes those predictions
at each checkpoint; the dashboard API reads them back and pairs them with the
Monte-Carlo ground truth. See docs/react_dashboard.md.
"""

from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

from scribblez.ffi import analyze_post_move_gcg

# The default frozen evaluation set: penultimate-bingo positions whose Monte-Carlo
# ground truth lives in monte-carlo-sim-results.json next to the GCGs.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "positions" / "NWL23" / "post-move-value-test-dataset"
GROUND_TRUTH_FILENAME = "monte-carlo-sim-results.json"

BOARD_SIZE = 15


def dataset_gcgs(dataset_dir: str | Path) -> list[Path]:
    """The dataset's GCG files in stable natural order (pos-1, pos-2, ..., pos-10)."""
    return natsorted(Path(dataset_dir).glob("*.gcg"), key=lambda p: p.name)


def load_inputs(dataset_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """(names, inputs): each position's flat post-move model-input tensor, stacked
    (N, F).

    `names` are the GCG file stems (the dashboard's position labels, matching the
    Monte-Carlo ground-truth keys). Each input is encoded from the POV of the player
    that made the final move -- the same seat the ground truth scores.
    """
    names: list[str] = []
    rows: list[np.ndarray] = []
    for gcg in dataset_gcgs(dataset_dir):
        names.append(gcg.stem)
        rows.append(analyze_post_move_gcg(gcg.read_text()))
    return names, (np.stack(rows) if rows else np.zeros((0, 0), np.float32))


def split_input(inputs: np.ndarray, spatial_planes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split flat inputs (N, F) into the model's (spatial (N, P, 15, 15), scalar
    (N, S)) halves -- the encoder lays spatial planes (channel-major) before the
    scalar block."""
    cells = BOARD_SIZE * BOARD_SIZE
    spatial = inputs[:, : spatial_planes * cells].reshape(
        -1, spatial_planes, BOARD_SIZE, BOARD_SIZE
    )
    scalar = inputs[:, spatial_planes * cells :]
    return spatial, scalar


@torch.no_grad()
def predict(model, inputs: np.ndarray, spatial_planes: int, device) -> dict:
    """Run `model` over the dataset inputs and decode each position's value outputs:

        wld      (N, 3) float32   softmax win/draw/loss probabilities (in that order)
        sd_mean  (N,)   float32   predicted final-score-delta mean (points)
        sd_std   (N,)   float32   predicted final-score-delta std (points, a Gaussian)

    These are exactly what the dashboard pairs against the Monte-Carlo ground truth:
    the WLD bars and the score-delta Gaussian overlaid on the MC histogram.
    """
    spatial, scalar = split_input(inputs, spatial_planes)
    sp = torch.from_numpy(np.ascontiguousarray(spatial)).to(device)
    sc = torch.from_numpy(np.ascontiguousarray(scalar)).to(device)
    out = model(sp, sc)
    wld = torch.softmax(out["wld"], dim=-1).cpu().numpy().astype(np.float32)
    sd = out["score_diff"].cpu().numpy().astype(np.float32)
    return {"wld": wld, "sd_mean": sd[:, 0], "sd_std": sd[:, 1]}
