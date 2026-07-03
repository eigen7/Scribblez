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

import json
import math
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

from scribblez.ffi import analyze_post_move_gcg

# The frozen evaluation sets: penultimate-bingo positions whose Monte-Carlo ground
# truth lives in monte-carlo-sim-results.json next to the GCGs. DEFAULT_DATASET is the
# small hand-built set (loose .gcg files) the Positions tab scrubs; LARGE_DATASET is
# the machine-harvested set (committed as part-*.gcgs bundles) the Loss tab's
# aggregate quality curves are measured over.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "positions" / "NWL23" / "post-move-value-test-dataset"
LARGE_DATASET = REPO_ROOT / "positions" / "NWL23" / "post-move-value-test-dataset-large"
GROUND_TRUTH_FILENAME = "monte-carlo-sim-results.json"

# The record boundary in a part-*.gcgs bundle: every GCG block starts with this line.
GCG_MARKER = "#character-encoding UTF-8"

BOARD_SIZE = 15


def dataset_gcgs(dataset_dir: str | Path) -> list[Path]:
    """The dataset's loose GCG files in stable natural order (pos-1, pos-2, ...)."""
    return natsorted(Path(dataset_dir).glob("*.gcg"), key=lambda p: p.name)


def split_bundle(text: str) -> list[str]:
    """Split a part-*.gcgs bundle's text into its GCG blocks (each starting at GCG_MARKER)."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line == GCG_MARKER and current:
            blocks.append("\n".join(current).rstrip() + "\n")
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).rstrip() + "\n")
    return blocks


def _dataset_items(dataset_dir: str | Path) -> list[tuple[str, str]]:
    """(stem, gcg_text) for every position in the dataset, in stable order.

    A dataset is either loose `pos-*.gcg` files (the small hand-built set) or
    `part-*.gcgs` bundles of concatenated GCG blocks (the large harvested set). For
    bundles the stems are `pos-NNNN` in bundle order, matching the Monte-Carlo
    ground-truth keys the build_post_move_test_set explode step produced.
    """
    loose = dataset_gcgs(dataset_dir)
    if loose:
        return [(gcg.stem, gcg.read_text()) for gcg in loose]
    items: list[tuple[str, str]] = []
    for bundle in natsorted(Path(dataset_dir).glob("part-*.gcgs"), key=lambda p: p.name):
        for block in split_bundle(bundle.read_text()):
            items.append((f"pos-{len(items) + 1:04d}", block))
    return items


def load_inputs(dataset_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """(names, inputs): each position's flat post-move model-input tensor, stacked
    (N, F).

    `names` are the position stems (matching the Monte-Carlo ground-truth keys). Each
    input is encoded from the POV of the player that made the final move -- the same
    seat the ground truth scores. Works for both loose-.gcg and bundled datasets.
    """
    items = _dataset_items(dataset_dir)
    names = [stem for stem, _ in items]
    rows = [analyze_post_move_gcg(text) for _, text in items]
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


def load_ground_truth(dataset_dir: str | Path, names: list[str]) -> dict:
    """Per-position Monte-Carlo ground truth, aligned to `names`.

    Reads monte-carlo-sim-results.json and returns arrays over the positions:
        win_eq (N,)   empirical win equity (win + 0.5*draw)
        wld    (N, 3) empirical [win, draw, loss] fractions (model output order)
        mean   (N,)   final-score-delta mean (points)
        std    (N,)   final-score-delta std (points)
    """
    gt = json.loads((Path(dataset_dir) / GROUND_TRUTH_FILENAME).read_text())
    n = len(names)
    win_eq = np.empty(n, np.float32)
    wld = np.empty((n, 3), np.float32)
    mean = np.empty(n, np.float32)
    std = np.empty(n, np.float32)
    for i, name in enumerate(names):
        entry = gt[name]
        total = entry["n"]
        w = entry["wld"]
        p_win, p_draw, p_loss = w["win"] / total, w["draw"] / total, w["loss"] / total
        wld[i] = (p_win, p_draw, p_loss)
        win_eq[i] = p_win + 0.5 * p_draw
        hist = {int(d): c for d, c in entry["score_delta_hist"].items()}
        count = sum(hist.values()) or 1
        m = sum(d * c for d, c in hist.items()) / count
        var = sum(c * (d - m) ** 2 for d, c in hist.items()) / count
        mean[i] = m
        std[i] = math.sqrt(max(var, 0.0))
    return {"win_eq": win_eq, "wld": wld, "mean": mean, "std": std}


def quality_metrics(preds: dict, gt: dict) -> dict:
    """Aggregate model-vs-Monte-Carlo quality scalars over the dataset (all lower is
    better): win-equity MAE and full-WLD Brier for the win/draw/loss head, and mean-
    and std-MAE (points) for the score-delta Gaussian head."""
    pred_win_eq = preds["wld"][:, 0] + 0.5 * preds["wld"][:, 1]
    return {
        "eval_win_mae": float(np.mean(np.abs(pred_win_eq - gt["win_eq"]))),
        "eval_wld_brier": float(np.mean(np.sum((preds["wld"] - gt["wld"]) ** 2, axis=1))),
        "eval_sd_mean_mae": float(np.mean(np.abs(preds["sd_mean"] - gt["mean"]))),
        "eval_sd_std_mae": float(np.mean(np.abs(preds["sd_std"] - gt["std"]))),
    }
