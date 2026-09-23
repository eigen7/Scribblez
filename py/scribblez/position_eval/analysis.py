"""The position evaluation eval sets: loading positions and Monte-Carlo ground
truth, running a model over them, and scoring it.

The trainer scores every checkpoint on the large set (quality_metrics,
placement_metrics); the dashboard's Positions tab shows the small set
(docs/react_dashboard.md).

Ground truth comes from monte_carlo_sim_tool, run offline and committed beside
each dataset: WLD counts, the final score-differential histogram, and placement
planes. There is one results file per information condition, i.e. what a
rollout knows of the opponent's leave (ground_truth_path). A model is scored
against the condition it trains under.

Each position is the board after its GCG's final move, evaluated from the POV of
the player who made that move.
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
from scribblez.paths import EVAL_POSITIONS_DIRS
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES

# DEFAULT_DATASET: the small hand-built set of loose .gcg files, shown in the
# Positions tab. LARGE_DATASET: a machine-harvested set stored as part-*.gcgs
# bundles, behind the Loss tab's quality curves.
DEFAULT_DATASET, LARGE_DATASET = EVAL_POSITIONS_DIRS

# Every GCG block in a part-*.gcgs bundle starts with this line.
GCG_MARKER = "#character-encoding UTF-8"

BOARD_SIZE = 15


def ground_truth_path(dataset_dir: str | Path, face_up_leaves: bool) -> Path:
    """The dataset's Monte-Carlo results file for an information condition."""
    condition = "face-up-leaves" if face_up_leaves else "hidden-leaves"
    return Path(dataset_dir) / f"monte-carlo-sim-results.{condition}.json"


def dataset_gcgs(dataset_dir: str | Path) -> list[Path]:
    """The dataset's loose GCG files in stable natural order (pos-1, pos-2, ...)."""
    return natsorted(Path(dataset_dir).glob("*.gcg"), key=lambda p: p.name)


def split_bundle(text: str) -> list[str]:
    """Split a part-*.gcgs bundle's text into its GCG blocks."""
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
    """(stem, gcg_text) for every position, in stable order, from either loose
    `pos-*.gcg` files or `part-*.gcgs` bundles. Bundle positions are named
    `pos-NNNN` in bundle order, the names scripts/build_position_eval_test_set.py
    gives them when it computes the ground truth.
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
    """(names, inputs): the position stems, which key the ground truth, and
    the flat model inputs under `arm`, stacked (N, F)."""
    items = _dataset_items(dataset_dir)
    names = [stem for stem, _ in items]
    rows = [analyze_position_eval_gcg(text, arm) for _, text in items]
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
    """Run `model` over the dataset inputs and decode its outputs:

        wld               (N, 3)     float32  win/draw/loss probabilities
        sd_mean           (N,)       float32  final score-differential mean (points)
        sd_std            (N,)       float32  final score-differential std (points)
        placement_logits  (N, 4, C)  float32  raw footprint logits, PLACEMENT_HEAD_NAMES order

    Placement logits stay raw; collapse_placement turns them into per-cell planes.
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
    """(texts, legal): what collapsing and scoring placement predictions needs
    beyond the model inputs. `texts` are the positions' GCGs, from which the
    engine rebuilds each board. `legal` (N, 4, 15, 15) bool marks the cells some
    legal footprint of each head covers."""
    items = _dataset_items(dataset_dir)
    texts = [text for _, text in items]
    legal = np.stack([legal_position_eval_placement(text) for text in texts])
    return texts, legal


def collapse_placement(logits: np.ndarray, texts: list[str]) -> np.ndarray:
    """Collapse placement logits (N, 4, C) into per-cell planes (N, 4, 15, 15)
    via the engine (ffi.collapse_position_eval_placement). These are the planes
    the Positions tab draws and the Monte-Carlo planes are compared with."""
    return np.stack(
        [
            collapse_position_eval_placement(text, raw)
            for raw, text in zip(logits, texts, strict=True)
        ]
    )


def load_ground_truth(dataset_dir: str | Path, names: list[str], face_up_leaves: bool) -> dict:
    """Monte-Carlo ground truth for an information condition, aligned to `names`:

    win_eq     (N,)            win + 0.5 * draw fraction
    wld        (N, 3)          [win, draw, loss] fractions
    mean       (N,)            final score-differential mean (points)
    std        (N,)            final score-differential std (points)
    placement  (N, 4, 15, 15)  per-cell rollout fractions in PLACEMENT_HEAD_NAMES
                               order: how often that seat's next move covered the
                               cell (and, for win heads, that seat won); None if the
                               results file has no placement planes
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
    """Model-vs-Monte-Carlo value metrics over the dataset, all lower-is-better:
    win-equity MAE, WLD Brier score, and score-differential mean and std MAE."""
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
    """Per-head placement metrics: the model's collapsed planes against the
    Monte-Carlo planes (both (N, 4, 15, 15)), restricted to each head's legal
    cells. The aggregate form of the Positions tab's residual heat map.

        eval_place_l1_<head>    sum of |model - MC| over the cells, averaged over
                                positions (lower is better). A plane sums to the
                                expected tiles placed (times the win probability
                                for a win head), so this is tiles' worth of
                                coverage on the wrong cells. It is absolute rather
                                than relative to the MC mass, which is near zero
                                for a win head whose seat rarely wins.
        eval_place_top1_<head>  fraction of positions where the model's and the
                                rollouts' most-covered cells agree

    Positions where a head's MC plane is empty (a win head whose seat never won)
    are skipped for that head; a head empty everywhere records nothing.

    The self heads are comparable only because the MC planes credit each
    reply's footprint as decoded on the position's board, the way the collapse
    does, rather than its literal squares (accumulate_rollout_placement in
    engine/include/sim/monte_carlo_sim.h).
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
