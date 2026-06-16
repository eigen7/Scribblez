"""Structural monotonicity probe (roadmap 3.1).

For each fixed position in the evaluation subset, the model's win rate
(Pr[win] + 0.5 * Pr[draw]) is read off across a sweep of score differentials.
A structurally coherent value model produces a curve that is monotonically
increasing in the active player's score advantage and sigmoid-shaped
(saturating near 0 when far behind and near 1 when far ahead). Each curve is
scored on three properties and rendered as one panel of a stacked grid image,
so successive generations can be scrolled to watch the curves become sigmoidal
as training progresses.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CurveScore:
    """Structural scores for a single win-rate curve."""

    monotonicity_violations: int  # count of strictly decreasing steps
    violation_fraction: float  # violations / (num_steps - 1)
    sigmoid_r2: float  # R^2 of a logistic fit to the curve
    total_variation: float  # sum of |finite differences|
    structural_score: float  # combined plausibility in [0, 1]


def _logistic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _sigmoid_r2(diffs: np.ndarray, curve: np.ndarray) -> float:
    """R^2 of the best-fit logistic curve; 0 if the fit fails to converge."""
    from scipy.optimize import curve_fit  # local import: optional at module load

    ss_tot = float(np.sum((curve - curve.mean()) ** 2))
    if ss_tot < 1e-9:  # flat curve -- a logistic explains none of its (zero) variance
        return 0.0
    try:
        # Scale x to keep the fit well-conditioned; p0 ~ a gentle slope at 0.
        x = diffs / 100.0
        popt, _ = curve_fit(_logistic, x, curve, p0=[1.0, 0.0], maxfev=10000)
        resid = curve - _logistic(x, *popt)
        return float(1.0 - np.sum(resid**2) / ss_tot)
    except (RuntimeError, ValueError):
        return 0.0


def score_curve(diffs: np.ndarray, curve: np.ndarray) -> CurveScore:
    """Score one win-rate curve on monotonicity, sigmoid fit, and smoothness."""
    deltas = np.diff(curve)
    violations = int(np.sum(deltas < 0.0))
    n_steps = max(len(deltas), 1)
    violation_fraction = violations / n_steps
    sigmoid_r2 = _sigmoid_r2(diffs, curve)
    total_variation = float(np.sum(np.abs(deltas)))
    # Coherence rewards a good sigmoid fit and penalizes non-monotonicity.
    structural_score = max(0.0, sigmoid_r2) * (1.0 - violation_fraction)
    return CurveScore(
        monotonicity_violations=violations,
        violation_fraction=violation_fraction,
        sigmoid_r2=sigmoid_r2,
        total_variation=total_variation,
        structural_score=structural_score,
    )


@dataclass
class MonotonicityReport:
    """Aggregate monotonicity-probe result over a whole bank."""

    scores: list[CurveScore]
    mean_structural_score: float
    mean_sigmoid_r2: float
    total_violations: int


def score_monotonicity(score_diffs: np.ndarray, win_rate: np.ndarray) -> MonotonicityReport:
    """Score every position's win-rate curve."""
    scores = [score_curve(score_diffs, win_rate[i]) for i in range(win_rate.shape[0])]
    return MonotonicityReport(
        scores=scores,
        mean_structural_score=float(np.mean([s.structural_score for s in scores])),
        mean_sigmoid_r2=float(np.mean([s.sigmoid_r2 for s in scores])),
        total_violations=int(np.sum([s.monotonicity_violations for s in scores])),
    )


def render_monotonicity(
    score_diffs: np.ndarray, win_rate: np.ndarray, report: MonotonicityReport, path, title: str
) -> None:
    """Render the per-position win-rate curves as a stacked grid image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = win_rate.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False)
    diffs = score_diffs

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off")
            continue
        sc = report.scores[i]
        ax.plot(diffs, win_rate[i], color="#1f77b4", lw=1.5)
        ax.axhline(0.5, color="0.8", lw=0.8, ls="--")
        ax.axvline(0.0, color="0.8", lw=0.8, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(diffs[0], diffs[-1])
        ax.set_title(f"#{i} R²={sc.sigmoid_r2:.2f} viol={sc.monotonicity_violations}", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{title}  |  mean struct={report.mean_structural_score:.3f} "
        f"mean R²={report.mean_sigmoid_r2:.3f} viol={report.total_violations}",
        fontsize=11,
    )
    fig.supxlabel("score differential (active − opponent)", fontsize=9)
    fig.supylabel("win rate  (Pr[win] + ½·Pr[draw])", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
