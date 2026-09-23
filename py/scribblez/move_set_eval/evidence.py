"""Builds the fusion stage's EvidenceInputs (scribblez.evidence_fusion) for
one position.

An evidence token combines a simmed candidate's move encoding (the same
moves.encode_moves rows candidates use), its raw sim observations (.sobs
records, via scribblez.sim_evidence.sobs), and the model's own evidence-free
predictions for it. Only the observations are stored: they do not depend on
any model, so they never go stale. The caller supplies the predictions as
slices of the plain first pass over the full candidate set, which it runs
anyway.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch

from scribblez.evidence_fusion import (
    NUM_EVIDENCE_PLANES,
    NUM_EVIDENCE_SCALARS,
    NUM_OBSERVED_PLANES,
    NUM_PREDICTED_PLANES,
    EvidenceInputs,
)
from scribblez.sim_evidence.sobs import BOARD, candidate_slot_planes, observed_slot_planes

from .model import footprint_slot_planes
from .moves import encode_moves, move_encoding_dims

# EVIDENCE_PLANE_NAMES' block boundaries: observed | predicted | candidate.
_PREDICTED_END = NUM_OBSERVED_PLANES + NUM_PREDICTED_PLANES


def observed_scalars(obs: np.ndarray) -> np.ndarray:
    """(K,) .sobs records -> (K, 6) observed evidence scalars: win/draw/loss
    frequencies, delta mean and std (score points / 100), and
    log1p(rollouts) / 8."""
    n = np.maximum(obs["n"].astype(np.float64), 1.0)
    delta_mean = obs["delta_sum"] / n
    delta_var = np.maximum(obs["delta_sq_sum"] / n - delta_mean**2, 0.0)
    cols = [
        obs["wins"] / n,
        obs["draws"] / n,
        obs["losses"] / n,
        delta_mean / 100.0,
        np.sqrt(delta_var) / 100.0,
        np.log1p(obs["n"]) / 8.0,
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def predicted_scalars(first_pass: dict[str, torch.Tensor]) -> np.ndarray:
    """First-pass outputs for the K evidence candidates -> (K, 5)
    [p_win, p_draw, p_loss, sd mean / 100, sd std / 100], scaled like the
    observed delta moments."""
    wld = torch.softmax(first_pass["wld"].detach(), dim=1).cpu().float().numpy()
    sd = first_pass["score_diff"].detach().cpu().float().numpy() / 100.0
    return np.concatenate([wld, sd], axis=1)


def _pad_rows(rows: np.ndarray, max_e: int) -> np.ndarray:
    padded = np.zeros((max_e, *rows.shape[1:]), dtype=rows.dtype)
    padded[: len(rows)] = rows
    return padded


def _row_tensor(
    rows: np.ndarray, max_e: int, cast: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """(K, ...) rows -> a zero-padded (1, max_e, ...) tensor."""
    return torch.from_numpy(_pad_rows(rows, max_e)).to(device=device, dtype=cast).unsqueeze(0)


def build_evidence_inputs(
    moves: np.ndarray,
    obs: np.ndarray,
    pre_move_diff: int,
    first_pass: dict[str, torch.Tensor],
    *,
    max_e: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> EvidenceInputs:
    """One position's evidence set as (1, max_e, ...) model inputs.

    `moves`/`obs` are the simmed candidates' .sobs rows (a SobsPosition's
    arrays or a prefix of them). `pre_move_diff` is the mover's score
    differential before the move. `first_pass` holds the model's
    evidence-free "wld", "score_diff" and "planes" for the same K candidates
    in the same order.
    """
    k = len(moves)
    if k > max_e:
        raise ValueError(f"evidence set of {k} candidates does not fit max_e={max_e}")
    if first_pass["planes"].shape[0] != k:
        raise ValueError("first-pass rows do not match the evidence candidates")
    if k == 0:
        return empty_evidence_inputs(max_e, dtype=dtype, device=device)

    enc = encode_moves(np.asarray(moves), np.full(k, pre_move_diff, dtype=np.int32))

    planes = np.zeros((k, NUM_EVIDENCE_PLANES, BOARD, BOARD), dtype=np.float32)
    planes[:, :NUM_OBSERVED_PLANES] = observed_slot_planes(obs)
    planes[:, NUM_OBSERVED_PLANES:_PREDICTED_END] = (
        footprint_slot_planes(first_pass["planes"].detach()).cpu().float().numpy()
    )
    planes[:, _PREDICTED_END:] = candidate_slot_planes(moves)

    scalars = np.concatenate([observed_scalars(obs), predicted_scalars(first_pass)], axis=1)
    assert scalars.shape[1] == NUM_EVIDENCE_SCALARS

    mask = np.zeros(max_e, dtype=bool)
    mask[:k] = True

    return EvidenceInputs(
        letters=_row_tensor(enc["letters"], max_e, torch.int64, device),
        blanks=_row_tensor(enc["blanks"], max_e, torch.int64, device),
        squares=_row_tensor(enc["squares"], max_e, torch.int64, device),
        tile_mask=_row_tensor(enc["tile_mask"], max_e, dtype, device),
        scalars=_row_tensor(enc["scalars"], max_e, dtype, device),
        obs_planes=_row_tensor(planes, max_e, dtype, device),
        obs_scalars=_row_tensor(scalars, max_e, dtype, device),
        mask=torch.from_numpy(mask).to(device=device).unsqueeze(0),
    )


def empty_evidence_inputs(
    max_e: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> EvidenceInputs:
    """The empty evidence set as (1, max_e, ...) inputs, every row masked out,
    under which the model computes exactly its plain pass."""
    max_tiles, num_scalars, _, _ = move_encoding_dims()
    zeros = functools.partial(_zero_rows, max_e, dtype=dtype, device=device)
    return EvidenceInputs(
        letters=zeros(max_tiles, dtype=torch.int64),
        blanks=zeros(max_tiles, dtype=torch.int64),
        squares=zeros(max_tiles, dtype=torch.int64),
        tile_mask=zeros(max_tiles),
        scalars=zeros(num_scalars),
        obs_planes=zeros(NUM_EVIDENCE_PLANES, BOARD, BOARD),
        obs_scalars=zeros(NUM_EVIDENCE_SCALARS),
        mask=zeros(dtype=torch.bool),
    )


def _zero_rows(max_e: int, *shape: int, dtype: torch.dtype, device) -> torch.Tensor:
    """A (1, max_e, *shape) zero tensor."""
    return torch.zeros(1, max_e, *shape, dtype=dtype, device=device)


def collate_evidence(items: list[EvidenceInputs]) -> EvidenceInputs:
    """Stack per-position (1, E, ...) EvidenceInputs into a (P, E, ...) batch.
    All items must share the same padded width E."""
    fields = {}
    for f in dataclasses.fields(EvidenceInputs):
        fields[f.name] = torch.cat([getattr(item, f.name) for item in items], dim=0)
    return EvidenceInputs(**fields)
