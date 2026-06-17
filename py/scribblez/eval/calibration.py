"""Calibration evaluation on the full held-out test set (roadmap 3.3).

Unlike the structural probes (which sweep a frozen 12-position subset), calibration
measures probabilistic accuracy over EVERY test position against its stored
ground-truth targets: does a predicted win rate of 70% correspond to actually
winning ~70% of the time, and does the score-diff head's predicted distribution
match the real final score differential?

Two heads are scored:
  * WLD -- win rate (Pr[win] + 0.5*Pr[draw], matching the monotonicity probe) vs
    the actual game value (1 / 0.5 / 0 for win / draw / loss). Reported as Brier
    score, multiclass log-loss (== the test-set wld cross-entropy), a decile
    reliability diagram, and its expected calibration error (ECE).
  * ScoreDiff -- the predicted distribution's mean vs the actual final score
    differential. Reported as mean absolute error, signed bias, and sharpness
    (mean predictive entropy), plus a reliability binning.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# WLD win-rate reliability uses deciles; score-diff reliability uses a fixed
# differential range so the panel-2 axes are comparable across generations.
NUM_WLD_BINS = 10
NUM_SD_BINS = 20
SD_BIN_RANGE = 200  # score-diff reliability x/y axis half-width


@dataclass
class CalibrationReport:
    """Calibration metrics over the full test set (binned aggregates only)."""

    num_positions: int

    # WLD (win-rate) calibration.
    brier: float
    log_loss: float
    ece: float
    rel_bin_edges: np.ndarray  # (NUM_WLD_BINS+1,) decile edges in [0, 1]
    rel_bin_pred: np.ndarray  # (NUM_WLD_BINS,) mean predicted win rate (nan if empty)
    rel_bin_actual: np.ndarray  # (NUM_WLD_BINS,) mean actual value (nan if empty)
    rel_bin_count: np.ndarray  # (NUM_WLD_BINS,) per-bin counts

    # Score-diff calibration.
    scorediff_mae: float
    scorediff_bias: float
    scorediff_sharpness: float  # mean predictive entropy (nats)
    scorediff_mean_std: float  # mean predictive stddev (score points)
    sd_bin_edges: np.ndarray  # (NUM_SD_BINS+1,) edges over [-RANGE, +RANGE]
    sd_bin_pred: np.ndarray  # (NUM_SD_BINS,) mean predicted mean (nan if empty)
    sd_bin_actual: np.ndarray  # (NUM_SD_BINS,) mean actual diff (nan if empty)
    sd_bin_count: np.ndarray  # (NUM_SD_BINS,) per-bin counts


def _bin_means(values: np.ndarray, actuals: np.ndarray, x: np.ndarray, edges: np.ndarray):
    """Mean of `values` and `actuals` within each [edges[i], edges[i+1]) bucket of `x`.

    Returns (mean_values, mean_actuals, counts); empty buckets are nan / 0.
    """
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    mean_v = np.full(n_bins, np.nan)
    mean_a = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        counts[b] = c
        if c:
            mean_v[b] = float(values[sel].mean())
            mean_a[b] = float(actuals[sel].mean())
    return mean_v, mean_a, counts


@torch.no_grad()
def evaluate_calibration(
    model,
    test_dataset,
    device: torch.device,
    *,
    batch_size: int = 512,
    seed: int = 0,
    max_positions: int | None = None,
) -> CalibrationReport:
    """Score `model`'s calibration over one full pass of `test_dataset`.

    Evaluated under eval() (BatchNorm in inference mode); the prior train/eval
    state is restored before returning. Per-position scalars are collected
    (a few float arrays of length N -- trivial for test sets up to millions).
    """
    centers = torch.arange(801, device=device, dtype=torch.float32) - 400.0

    was_training = model.training
    model.eval()
    win_rate, value, wld_logprob = [], [], []
    pred_mean, actual_diff, entropy, pdf_std = [], [], [], []
    seen = 0
    try:
        for batch in test_dataset.iter_batches(batch_size, seed=seed):
            spatial = batch["input_spatial"].to(device)
            scalar = batch["input_scalar"].to(device)
            wld_t = batch["wld"].to(device)
            sd_t = batch["score_diff"].to(device)
            out = model(spatial, scalar)

            wld_p = torch.softmax(out["wld"], dim=1)  # [win, draw, loss]
            win_rate.append((wld_p[:, 0] + 0.5 * wld_p[:, 1]).cpu().numpy())
            value.append((wld_t[:, 0] + 0.5 * wld_t[:, 1]).cpu().numpy())
            wld_logprob.append((-(wld_t * F.log_softmax(out["wld"], dim=1)).sum(1)).cpu().numpy())

            sd_p = torch.softmax(out["score_diff"], dim=1)
            mean = (sd_p * centers).sum(1)
            pred_mean.append(mean.cpu().numpy())
            actual_diff.append((sd_t.argmax(1).float() - 400.0).cpu().numpy())
            entropy.append((-(sd_p * torch.log(sd_p + 1e-12)).sum(1)).cpu().numpy())
            var = (sd_p * (centers[None, :] - mean[:, None]) ** 2).sum(1)
            pdf_std.append(torch.sqrt(var).cpu().numpy())

            seen += spatial.shape[0]
            if max_positions is not None and seen >= max_positions:
                break
    finally:
        if was_training:
            model.train()

    win_rate = np.concatenate(win_rate)
    value = np.concatenate(value)
    wld_logprob = np.concatenate(wld_logprob)
    pred_mean = np.concatenate(pred_mean)
    actual_diff = np.concatenate(actual_diff)
    entropy = np.concatenate(entropy)
    pdf_std = np.concatenate(pdf_std)
    n = len(win_rate)

    # WLD reliability + ECE.
    rel_edges = np.linspace(0.0, 1.0, NUM_WLD_BINS + 1)
    rel_pred, rel_actual, rel_count = _bin_means(win_rate, value, win_rate, rel_edges)
    nonempty = rel_count > 0
    ece = float(
        np.sum(rel_count[nonempty] / n * np.abs(rel_pred[nonempty] - rel_actual[nonempty]))
    )

    # Score-diff reliability over a fixed range (out-of-range clamps to end bins).
    sd_edges = np.linspace(-SD_BIN_RANGE, SD_BIN_RANGE, NUM_SD_BINS + 1)
    sd_pred, sd_actual, sd_count = _bin_means(pred_mean, actual_diff, pred_mean, sd_edges)

    return CalibrationReport(
        num_positions=n,
        brier=float(np.mean((win_rate - value) ** 2)),
        log_loss=float(np.mean(wld_logprob)),
        ece=ece,
        rel_bin_edges=rel_edges,
        rel_bin_pred=rel_pred,
        rel_bin_actual=rel_actual,
        rel_bin_count=rel_count,
        scorediff_mae=float(np.mean(np.abs(pred_mean - actual_diff))),
        scorediff_bias=float(np.mean(pred_mean - actual_diff)),
        scorediff_sharpness=float(np.mean(entropy)),
        scorediff_mean_std=float(np.mean(pdf_std)),
        sd_bin_edges=sd_edges,
        sd_bin_pred=sd_pred,
        sd_bin_actual=sd_actual,
        sd_bin_count=sd_count,
    )
