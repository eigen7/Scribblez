"""Shared forward pass for the structural probes.

Both probes sweep the same frozen positions over a score-differential range and
read different model heads, so a single batched inference feeds both. The
positions live in a standalone .slog (see sampling.build_test_subset); each
generation this re-encodes them via the FFI score-diff sweep and runs the
current model -- the model is the only thing that changes across generations.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scribblez.ffi import encode_score_diff_sweep, get_input_shapes, read_file_header


@dataclass
class ProbeOutputs:
    """Per-position, per-score-diff model readouts over the evaluation subset.

    Attributes:
        score_diffs: (R,) the swept input differentials (active - opponent).
        win_rate:    (N, R) Pr[win] + 0.5 * Pr[draw] from the WLD head.
        score_mean:  (N, R) predicted final-differential mean from the score-diff head.
        score_std:   (N, R) predicted final-differential std from the score-diff head.
    """

    score_diffs: np.ndarray
    win_rate: np.ndarray
    score_mean: np.ndarray
    score_std: np.ndarray

    @property
    def num_positions(self) -> int:
        return self.win_rate.shape[0]


@torch.no_grad()
def evaluate_subset(
    model,
    slog_path: str | Path,
    device: torch.device,
    diff_lo: int = -100,
    diff_hi: int = 100,
    post_move: bool = True,
) -> ProbeOutputs:
    """Run `model` over every (position, score-diff) cell of the subset .slog.

    Evaluated under eval() (BatchNorm in inference mode); the prior train/eval
    state is restored before returning.
    """
    num_pos, _ = read_file_header(slog_path)
    diffs = np.arange(diff_lo, diff_hi + 1, dtype=np.float32)
    r = diffs.shape[0]

    spatial_shape = next(s.dims for s in get_input_shapes() if s.name == "input_spatial")
    spatial_floats = int(np.prod(spatial_shape))

    # One FFI call for all positions: (N*R, input_floats), position-major.
    rows = encode_score_diff_sweep(slog_path, -1, diff_lo, diff_hi, post_move=post_move)
    rows = rows.reshape(num_pos, r, -1)

    was_training = model.training
    model.eval()
    win_rate = np.empty((num_pos, r), dtype=np.float32)
    score_mean = np.empty((num_pos, r), dtype=np.float32)
    score_std = np.empty((num_pos, r), dtype=np.float32)
    for k in range(num_pos):
        spatial = torch.from_numpy(rows[k, :, :spatial_floats].reshape(r, *spatial_shape)).to(device)
        scalar = torch.from_numpy(rows[k, :, spatial_floats:]).to(device)
        out = model(spatial, scalar)
        wld = torch.softmax(out["wld"], dim=1)  # [win, draw, loss]
        win_rate[k] = (wld[:, 0] + 0.5 * wld[:, 1]).cpu().numpy()
        sd = out["score_diff"]  # (r, 2): [mean, std]
        score_mean[k] = sd[:, 0].cpu().numpy()
        score_std[k] = sd[:, 1].cpu().numpy()
    if was_training:
        model.train()

    return ProbeOutputs(
        score_diffs=diffs, win_rate=win_rate, score_mean=score_mean, score_std=score_std
    )
