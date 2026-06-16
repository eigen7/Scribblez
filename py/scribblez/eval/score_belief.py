"""Score-belief probe (visualizing the score-diff head over a score sweep).

The model's score-diff head predicts a distribution over the final score
differential (an 801-bin PDF over a clipped range, KataGo-style). For each
fixed position in the evaluation subset, this probe sweeps the *input* score differential
and renders, for every input value, percentile bands (5/25/50/75/95) of the
predicted *final* score-diff distribution. The result is a fan chart per
position: x is the current score advantage, y is the believed final score
advantage, and the shaded bands show the spread of that belief.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _bin_centers(num_bins: int) -> np.ndarray:
    """Score-delta value of each bin: symmetric around 0, clip = (B-1)/2."""
    clip = (num_bins - 1) // 2
    return np.arange(num_bins, dtype=np.float64) - clip


def percentile_bands(score_pdf: np.ndarray, quantiles=DEFAULT_QUANTILES) -> np.ndarray:
    """Per-slice score-delta quantiles of a (..., B) PDF array.

    Returns an array shaped (..., len(quantiles)) holding the score-delta value
    at each requested cumulative probability, linearly interpolated across the
    bin CDF.
    """
    centers = _bin_centers(score_pdf.shape[-1])
    flat = score_pdf.reshape(-1, score_pdf.shape[-1])
    qs = np.asarray(quantiles, dtype=np.float64)
    out = np.empty((flat.shape[0], qs.shape[0]), dtype=np.float64)
    for i in range(flat.shape[0]):
        cdf = np.cumsum(flat[i])
        cdf /= cdf[-1]  # guard against tiny softmax normalization drift
        out[i] = np.interp(qs, cdf, centers)
    return out.reshape(*score_pdf.shape[:-1], qs.shape[0])


def render_score_belief(
    score_diffs: np.ndarray, score_pdf: np.ndarray, path, title: str, quantiles=DEFAULT_QUANTILES
) -> None:
    """Render per-position score-belief fan charts as a stacked grid image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = percentile_bands(score_pdf, quantiles)  # (N, S, Q)
    n = score_pdf.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False)
    diffs = score_diffs
    mid = len(quantiles) // 2  # index of the median quantile

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off")
            continue
        b = bands[i]  # (S, Q)
        # Shade symmetric quantile pairs from outermost (lightest) inward.
        for lo in range(mid):
            hi = len(quantiles) - 1 - lo
            ax.fill_between(diffs, b[:, lo], b[:, hi], color="#1f77b4", alpha=0.18 + 0.18 * lo)
        ax.plot(diffs, b[:, mid], color="#08306b", lw=1.3)  # median
        ax.plot(diffs, diffs, color="0.7", lw=0.8, ls="--")  # identity reference
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.set_xlim(diffs[0], diffs[-1])
        ax.set_title(f"#{i}", fontsize=8)
        ax.tick_params(labelsize=6)

    qpct = "/".join(f"{int(q * 100)}" for q in quantiles)
    fig.suptitle(f"{title}  |  score belief, {qpct} percentiles (dashed = identity)", fontsize=11)
    fig.supxlabel("input score differential (active − opponent)", fontsize=9)
    fig.supylabel("predicted final score differential", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
