"""The leaf eval model's held-out read, beside the win table's on the same rows.

The win table is what DeepBot's sims value a leaf with today (docs/plans/
deep_bot.md, "The leaf: a win table"), so it is the baseline the model has to
beat. Both are scored by the log loss of the mover's expected score (a win 1,
a draw 1/2) against the outcome, overall and by pool size. The model's
three-way WLD cross-entropy and its score-differential error are recorded too.

The table reads the leaf as the side to move (the opponent, who replies next)
sees it once both refills are drawn: their spread plus their leave's static
value, and the tiles they cannot see, the bag plus the mover's rack. That is
the cell the table was fitted on for the same position in the game
(sim/win_table_feature.cpp).
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from scribblez import ffi
from scribblez.deep_bot_leaf_eval.model import LAYOUT

RACK_SIZE = 7
SCORE_DIFF_SCALE = 100.0  # the row's score differential scale (input_encoder.h)
# Pool-size bands the log loss is broken down by: (name, lowest, highest).
POOL_BANDS = (("0_20", 0, 20), ("21_40", 21, 40), ("41_60", 41, 60), ("61_100", 61, 100))
EPS = 1e-6


def _seat_size(scalar: np.ndarray, offset: str) -> np.ndarray:
    """Per row, the tiles in the seat whose counts start at LAYOUT[offset]."""
    return scalar[:, LAYOUT[offset] : LAYOUT[offset] + 27].sum(axis=1).round().astype(np.int64)


def _seat_sizes(scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(my leave, the opponent's, the pool) sizes per row."""
    return (
        _seat_size(scalar, "my_leave0"),
        _seat_size(scalar, "opp_leave0"),
        _seat_size(scalar, "pool0"),
    )


def win_table_features(scalar: np.ndarray) -> np.ndarray:
    """(B, row scalars) -> WinTableFeature records from the side to move's view."""
    my_size, opp_size, pool = _seat_sizes(scalar)
    my_spread = np.rint(scalar[:, LAYOUT["score_diff"]] * SCORE_DIFF_SCALE)
    opp_value = (
        scalar[:, LAYOUT["opp_leave_value"]].astype(np.float64) * LAYOUT["leave_value_scale"]
    )
    cell_spread = -my_spread + opp_value
    their_refill = np.minimum(RACK_SIZE - opp_size, pool)
    features = np.zeros(len(scalar), dtype=ffi.struct_dtype("WinTableFeature"))
    # std::lround: half away from zero.
    features["spread"] = np.sign(cell_spread) * np.floor(np.abs(cell_spread) + 0.5)
    features["tiles_unseen"] = pool - their_refill + my_size
    return features


def expected_score_logloss(expected: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Per-row log loss of an expected score in [0, 1] against the outcome
    (1, 1/2 or 0)."""
    e = np.clip(expected, EPS, 1 - EPS)
    return -(outcome * np.log(e) + (1 - outcome) * np.log(1 - e))


def _model_rows(model, raw: dict, device) -> dict[str, np.ndarray]:
    out = model(raw["input_spatial"].to(device), raw["input_scalar"].to(device))
    probs = torch.softmax(out["wld"].float(), dim=1).cpu().numpy()
    wld = raw["wld"].numpy()
    return {
        "expected": probs[:, 0] + 0.5 * probs[:, 1],
        "wld_nll": -np.log(np.clip((probs * wld).sum(axis=1), EPS, 1.0)),
        "sd_error": np.abs(
            out["score_diff"][:, 0].float().cpu().numpy() - raw["score_diff"][:, 0].numpy()
        ),
        "outcome": wld[:, 0] + 0.5 * wld[:, 1],
        "pool": _seat_sizes(raw["input_scalar"].numpy())[2],
        "features": win_table_features(raw["input_scalar"].numpy()),
    }


def _table_expected(path: Path, features: np.ndarray, macondo_format: bool) -> np.ndarray:
    """The mover's expected score under the table: 1 - the side to move's."""
    return 1.0 - ffi.win_table_eval(path, features, macondo_format=macondo_format)


@torch.no_grad()
def evaluate(
    model, batches: Iterable[dict], device, win_table: Path, macondo_table: Path
) -> dict[str, float]:
    """The held-out metrics over every row of `batches`: the model's, the
    fitted `win_table`'s, and Macondo's table's."""
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    for raw in batches:
        for key, value in _model_rows(model, raw, device).items():
            parts.setdefault(key, []).append(value)
    rows = {k: np.concatenate(v) for k, v in parts.items()}
    expected = {
        "model": rows["expected"],
        "wintable": _table_expected(win_table, rows["features"], macondo_format=False),
        "wintable_macondo": _table_expected(macondo_table, rows["features"], macondo_format=True),
    }
    return eval_metrics(rows, expected)


def eval_metrics(rows: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, float]:
    m = {
        "eval_wld_nll": float(rows["wld_nll"].mean()),
        "eval_sd_mean_mae": float(rows["sd_error"].mean()),
    }
    for name, e in expected.items():
        loss = expected_score_logloss(e, rows["outcome"])
        m[f"eval_{name}_logloss"] = float(loss.mean())
        for band, lo, hi in POOL_BANDS:
            sel = (rows["pool"] >= lo) & (rows["pool"] <= hi)
            if sel.any():
                m[f"eval_{name}_logloss_pool_{band}"] = float(loss[sel].mean())
    return m
