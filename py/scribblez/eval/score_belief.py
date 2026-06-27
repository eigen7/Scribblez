"""Score-belief probe (visualizing the score-diff head over a score sweep).

The model's score-diff head predicts a Gaussian over the final score
differential: a mean and standard deviation. For each fixed position in the
evaluation subset, this probe sweeps the *input* score differential and computes,
for every input value, percentile bands (5/25/50/75/95) of that predicted
Gaussian. The dashboard renders these as a fan chart per position: x is the
current score advantage, y is the believed final score advantage, and the shaded
bands show the spread of that belief.
"""

import statistics

import numpy as np

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def gaussian_bands(mean: np.ndarray, std: np.ndarray, quantiles=DEFAULT_QUANTILES) -> np.ndarray:
    """Per-element score-delta quantiles of a Gaussian (mean, std) belief.

    `mean` and `std` are broadcast-compatible arrays of identical shape (...,).
    Returns an array shaped (..., len(quantiles)) holding the score-delta value
    at each requested cumulative probability, i.e. mean + std * z_q where z_q is
    the standard-normal quantile for q.
    """
    z = np.array(
        [statistics.NormalDist().inv_cdf(q) for q in quantiles], dtype=np.float64
    )
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return mean[..., None] + std[..., None] * z
