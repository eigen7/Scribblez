"""Score-belief probe (visualizing the score-diff head over a score sweep).

The model's score-diff head predicts a distribution over the final score
differential (an 801-bin PDF over a clipped range, KataGo-style). For each
fixed position in the evaluation subset, this probe sweeps the *input* score differential
and computes, for every input value, percentile bands (5/25/50/75/95) of the
predicted *final* score-diff distribution. The dashboard renders these as a fan
chart per position: x is the current score advantage, y is the believed final
score advantage, and the shaded bands show the spread of that belief.
"""


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
