"""Run the evaluation suites and persist their results to the dashboard DB.

Shared by train.py (per-epoch) and evaluate.py (one-off on a checkpoint): each
helper computes an eval and writes its data to the per-tag SQLite store, returning
the scalar metrics for the caller to fold into the `metrics` table.
"""

import sys
from pathlib import Path

import numpy as np

from scribblez.dashboard import db

from .calibration import evaluate_calibration
from .monotonicity import score_monotonicity
from .probe_eval import evaluate_subset
from .score_belief import DEFAULT_QUANTILES, gaussian_bands
from .web_render import render_position_images


def render_boards(slog_path: str | Path, image_dir: str | Path):
    """Render the test-subset board PNGs (best-effort) for the dashboard to display.

    The boards are static across generations, so this runs once. Failure (no
    Node/Playwright toolchain) is non-fatal -- the dashboard simply shows no
    board for those positions.
    """
    try:
        render_position_images(slog_path, image_dir)
    except Exception as e:  # noqa: BLE001 -- board rendering is best-effort
        print(
            f"WARNING: board image render failed ({e}); dashboard will show no boards.",
            file=sys.stderr,
        )


def run_probes(model, slog_path, device, conn, epoch: int, diff_lo: int, diff_hi: int) -> dict:
    """Run the structural probes, store their data, and return the aggregate metrics."""
    outs = evaluate_subset(model, slog_path, device, diff_lo=diff_lo, diff_hi=diff_hi)
    report = score_monotonicity(outs.score_diffs, outs.win_rate)
    curve_scores = np.array(
        [[s.structural_score, s.sigmoid_r2, s.monotonicity_violations] for s in report.scores],
        dtype=np.float32,
    )
    db.write_monotonicity(conn, epoch, outs.score_diffs, outs.win_rate, curve_scores)
    bands = gaussian_bands(outs.score_mean, outs.score_std, DEFAULT_QUANTILES).astype(np.float32)
    quantiles = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
    db.write_score_belief(conn, epoch, outs.score_diffs, quantiles, bands)
    print(
        f"  probe: mean struct={report.mean_structural_score:.3f} "
        f"mean R²={report.mean_sigmoid_r2:.3f} viol={report.total_violations}"
    )
    return {
        "probe_mean_structural_score": report.mean_structural_score,
        "probe_mean_sigmoid_r2": report.mean_sigmoid_r2,
        "probe_total_violations": report.total_violations,
    }


def run_calibration(model, test_ds, device, conn, epoch: int, batch_size: int = 512) -> dict:
    """Score calibration over the full test set, store its binning, return scalar metrics."""
    report = evaluate_calibration(model, test_ds, device, batch_size=batch_size)
    db.write_calibration(conn, epoch, report)
    print(
        f"  calib: brier={report.brier:.4f} logloss={report.log_loss:.4f} ece={report.ece:.4f} "
        f"sd_mae={report.scorediff_mae:.1f} sd_bias={report.scorediff_bias:+.1f}"
    )
    return {
        "calib_brier": report.brier,
        "calib_log_loss": report.log_loss,
        "calib_ece": report.ece,
        "calib_scorediff_mae": report.scorediff_mae,
        "calib_scorediff_bias": report.scorediff_bias,
        "calib_scorediff_sharpness": report.scorediff_sharpness,
    }
