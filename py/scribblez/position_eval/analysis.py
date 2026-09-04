"""Position evaluation analysis: the GCG position dataset and decoding a model's
predictions over it for the dashboard.

Ground truth (per-position win/loss/draw + the exact final-score-delta distribution)
is precomputed offline by the `monte_carlo_sim_tool` and committed alongside the
dataset, one file per information condition (what a rollout knows of the
opponent's leave): `monte-carlo-sim-results.face-up-leaves.json` and
`monte-carlo-sim-results.hidden-leaves.json`. A model is measured against the
truth of the condition it trains under. This module owns the Python side: listing
the dataset's GCG positions, building the model-input batch from them (via
`scribblez.ffi.analyze_position_eval_gcg`), and decoding a model's outputs into the
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

from scribblez.ffi import (
    InputArm,
    analyze_position_eval_gcg,
    collapse_position_eval_placement,
    legal_position_eval_placement,
)
from scribblez.paths import REPO_ROOT
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES

# The frozen evaluation sets: post-move positions (the final recorded move is
# the evaluated player's) whose Monte-Carlo ground truth lives next to the GCGs.
# DEFAULT_DATASET is the small hand-built set (loose .gcg files) the Positions
# tab scrubs; LARGE_DATASET is the machine-harvested penultimate-bingo set
# (committed as part-*.gcgs bundles) the Loss tab's aggregate quality curves are
# measured over.
DEFAULT_DATASET = REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset"
LARGE_DATASET = REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset-large"

# The record boundary in a part-*.gcgs bundle: every GCG block starts with this line.
GCG_MARKER = "#character-encoding UTF-8"

BOARD_SIZE = 15


def ground_truth_path(dataset_dir: str | Path, face_up_leaves: bool) -> Path:
    """The dataset's Monte-Carlo results file for an information condition (the
    names monte_carlo_sim_tool writes)."""
    condition = "face-up-leaves" if face_up_leaves else "hidden-leaves"
    return Path(dataset_dir) / f"monte-carlo-sim-results.{condition}.json"


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
    ground-truth keys the build_position_eval_test_set explode step produced.
    """
    loose = dataset_gcgs(dataset_dir)
    if loose:
        return [(gcg.stem, gcg.read_text()) for gcg in loose]
    items: list[tuple[str, str]] = []
    for bundle in natsorted(Path(dataset_dir).glob("part-*.gcgs"), key=lambda p: p.name):
        for block in split_bundle(bundle.read_text()):
            items.append((f"pos-{len(items) + 1:04d}", block))
    return items


def load_inputs(dataset_dir: str | Path, arm: InputArm) -> tuple[list[str], np.ndarray]:
    """(names, inputs): each position's flat position-eval model-input tensor
    under `arm`, stacked (N, F).

    `names` are the position stems (matching the Monte-Carlo ground-truth keys). Each
    input is encoded from the POV of the player that made the final move -- the same
    seat the ground truth scores. Works for both loose-.gcg and bundled datasets.
    """
    items = _dataset_items(dataset_dir)
    names = [stem for stem, _ in items]
    rows = [analyze_position_eval_gcg(text, arm) for _, text in items]
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
    """Run `model` over the dataset inputs and decode each position's outputs:

        wld               (N, 3) float32   softmax win/draw/loss probabilities (in that order)
        sd_mean           (N,)   float32   predicted final-score-delta mean (points)
        sd_std            (N,)   float32   predicted final-score-delta std (points, a Gaussian)
        placement_logits  (N, 4, C) float32  the placement heads' raw footprint logits,
                                             in PLACEMENT_HEAD_NAMES order

    The value outputs are exactly what the dashboard pairs against the Monte-Carlo
    ground truth: the WLD bars and the score-delta Gaussian overlaid on the MC
    histogram. The placement logits are raw because masking and collapsing them
    to per-cell planes is the engine's job (collapse_placement).
    """
    spatial, scalar = split_input(inputs, spatial_planes)
    sp = torch.from_numpy(np.ascontiguousarray(spatial)).to(device)
    sc = torch.from_numpy(np.ascontiguousarray(scalar)).to(device)
    out = model(sp, sc)
    wld = torch.softmax(out["wld"], dim=-1).cpu().numpy().astype(np.float32)
    sd = out["score_diff"].cpu().numpy().astype(np.float32)
    logits = torch.stack([out[head] for head in PLACEMENT_HEAD_NAMES], dim=1)
    return {
        "wld": wld,
        "sd_mean": sd[:, 0],
        "sd_std": sd[:, 1],
        "placement_logits": logits.cpu().numpy().astype(np.float32),
    }


def load_placement_frame(dataset_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """What collapsing and scoring a dataset's placement predictions needs beyond
    the model inputs: each position's GCG text (the engine re-derives the board
    from it to mask and scatter the footprints) and its per-head cell legality,
    (N, 4, 15, 15) bool -- the cells some legal footprint of that head covers, the
    only cells a residual can live on."""
    items = _dataset_items(dataset_dir)
    texts = [text for _, text in items]
    legal = np.stack([legal_position_eval_placement(text) for text in texts])
    return texts, legal


def collapse_placement(logits: np.ndarray, texts: list[str]) -> np.ndarray:
    """The per-cell occupancy planes, (N, 4, 15, 15), of the placement logits
    (N, 4, C) over the positions' GCG `texts`: per position, the engine masks the
    illegal footprints, softmaxes, and scatters each footprint's probability onto
    the cells it covers -- Pr[the next move covers cell] for the plays heads,
    Pr[covers cell AND that seat wins] for the win heads. The same collapse the
    Positions tab draws and the Monte-Carlo planes count."""
    return np.stack(
        [
            collapse_position_eval_placement(text, raw)
            for raw, text in zip(logits, texts, strict=True)
        ]
    )


def load_ground_truth(dataset_dir: str | Path, names: list[str], face_up_leaves: bool) -> dict:
    """Per-position Monte-Carlo ground truth under an information condition,
    aligned to `names`.

    Reads the condition's results file and returns arrays over the positions:
        win_eq    (N,)   empirical win equity (win + 0.5*draw)
        wld       (N, 3) empirical [win, draw, loss] fractions (model output order)
        mean      (N,)   final-score-delta mean (points)
        std       (N,)   final-score-delta std (points)
        placement (N, 4, 15, 15) per-cell rollout fractions, PLACEMENT_HEAD_NAMES
                  order (the sim's PlacementCounts / n: how often that seat's
                  first move covered the cell, and did so in a rollout it won);
                  None when the results file predates the planes
    """
    gt = json.loads(ground_truth_path(dataset_dir, face_up_leaves).read_text())
    n = len(names)
    win_eq = np.empty(n, np.float32)
    wld = np.empty((n, 3), np.float32)
    mean = np.empty(n, np.float32)
    std = np.empty(n, np.float32)
    placement = np.empty((n, len(PLACEMENT_HEAD_NAMES), BOARD_SIZE, BOARD_SIZE), np.float32)
    has_placement = True
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
        planes = entry.get("placement")
        has_placement = has_placement and planes is not None
        if has_placement:
            for h, head in enumerate(PLACEMENT_HEAD_NAMES):
                placement[i, h] = np.asarray(planes[head], np.float32) / (total or 1)
    return {
        "win_eq": win_eq,
        "wld": wld,
        "mean": mean,
        "std": std,
        "placement": placement if has_placement else None,
    }


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


def placement_head_short(head: str) -> str:
    """'opp_next_placement' -> 'opp_next' (the metric-name suffix)."""
    return head.removesuffix("_placement")


def placement_metric_names() -> list[str]:
    """Every scalar `placement_metrics` records, grouped by statistic."""
    return [
        f"eval_place_{stat}_{placement_head_short(head)}"
        for stat in ("l1", "top1")
        for head in PLACEMENT_HEAD_NAMES
    ]


def placement_metrics(planes: np.ndarray, truth: np.ndarray, legal: np.ndarray) -> dict:
    """Aggregate placement quality vs Monte-Carlo over the dataset, per head, from
    the model's collapsed planes and the MC planes (both (N, 4, 15, 15)) over the
    head's legal cells (N, 4, 15, 15) bool -- the systematic form of the Positions
    tab's residual heat map:

        eval_place_l1_<head>    misplaced coverage, in tiles: sum |model - MC| over
                                the legal cells, per position, averaged (lower is
                                better). A plane sums to the expected number of
                                tiles the move places (times the win probability
                                for a win head), so this is how many tiles' worth
                                of coverage sit on the wrong cells. Absolute
                                rather than relative to the MC mass, which is
                                near zero for a win head wherever that seat
                                rarely wins and would blow the ratio up.
        eval_place_top1_<head>  fraction of positions where the model's most
                                covered cell is the rollouts' most covered cell

    Positions where the MC plane is empty for a head (the win heads, when that
    seat never won a rollout) contribute to neither statistic for that head; a
    head empty on every position records nothing.
    """
    record = {}
    for h, head in enumerate(PLACEMENT_HEAD_NAMES):
        short = placement_head_short(head)
        p = np.where(legal[:, h], planes[:, h], 0.0).reshape(len(planes), -1)
        t = np.where(legal[:, h], truth[:, h], 0.0).reshape(len(truth), -1)
        scored = t.sum(axis=1) > 0
        if not scored.any():
            continue
        l1 = np.abs(p - t).sum(axis=1)[scored]
        top1 = p.argmax(axis=1)[scored] == t.argmax(axis=1)[scored]
        record[f"eval_place_l1_{short}"] = float(l1.mean())
        record[f"eval_place_top1_{short}"] = float(top1.mean())
    return record
