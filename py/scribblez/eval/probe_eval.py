"""Shared forward pass for the structural probes.

Both probes sweep the same frozen positions over a score-differential range and
read different model heads, so a single batched inference feeds both. The
positions live in a standalone .slog (see sampling.build_test_subset); each
generation this re-encodes them via the FFI score-diff sweep and runs the
current model -- the model is the only thing that changes across generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..ffi import dump_position, encode_score_diff_sweep, get_input_shapes, read_file_header


@dataclass
class ProbeOutputs:
    """Per-position, per-score-diff model readouts over the evaluation subset.

    Attributes:
        score_diffs: (R,) the swept input differentials (active - opponent).
        win_rate:    (N, R) Pr[win] + 0.5 * Pr[draw] from the WLD head.
        score_pdf:   (N, R, B) softmax distribution from the score-diff head.
    """

    score_diffs: np.ndarray
    win_rate: np.ndarray
    score_pdf: np.ndarray

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
    pdf_rows = []
    for k in range(num_pos):
        spatial = torch.from_numpy(rows[k, :, :spatial_floats].reshape(r, *spatial_shape)).to(device)
        scalar = torch.from_numpy(rows[k, :, spatial_floats:]).to(device)
        out = model(spatial, scalar)
        wld = torch.softmax(out["wld"], dim=1)  # [win, draw, loss]
        win_rate[k] = (wld[:, 0] + 0.5 * wld[:, 1]).cpu().numpy()
        pdf_rows.append(torch.softmax(out["score_diff"], dim=1).cpu().numpy())
    if was_training:
        model.train()

    return ProbeOutputs(score_diffs=diffs, win_rate=win_rate, score_pdf=np.stack(pdf_rows))


def write_position_dumps(slog_path: str | Path, directory: str | Path, post_move: bool = True) -> None:
    """Write each subset position's ASCII description to pos-NN.txt in `directory`.

    File pos-NN.txt corresponds to panel #NN in the probe analysis images.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    num_pos, _ = read_file_header(slog_path)
    for k in range(num_pos):
        (directory / f"pos-{k:02d}.txt").write_text(dump_position(slog_path, k, post_move=post_move))
