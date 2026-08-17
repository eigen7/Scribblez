"""Evidence inputs for the move set evaluation model's fusion stage.

Assembles EvidenceInputs (scribblez.evidence_fusion) for one position from
the two halves an evidence token carries: the raw sim observations of the
simmed candidates -- .sobs records (sim_observation_log.h), read via
scribblez.sim_evidence.sobs -- and the model's own evidence-free first-pass
outputs for the same candidates. Raw observations are model-independent and
never go stale; the prediction half is recomputed live by the caller (its
rows are slices of the full-candidate-set first pass it runs anyway), so no
stored artifact carries model outputs.

The move half reuses the engine's move encoding (moves.encode_moves), so the
evidence tokens describe their moves exactly the way candidate rows do.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch

from scribblez.evidence_fusion import (
    NUM_EVIDENCE_PLANES,
    NUM_EVIDENCE_SCALARS,
    EvidenceInputs,
)
from scribblez.sim_evidence.sobs import BOARD, move_footprint

from .moves import encode_moves, move_encoding_dims

# SimObservation's count-plane fields, in the placement-head order the
# observed half of EVIDENCE_PLANE_NAMES mirrors.
_COUNT_PLANES = ("opp_next_count", "self_next_count", "opp_win_count", "self_win_count")


def _observed_planes(moves: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """(K,) .sobs records -> (K, 5, 15, 15): the four count planes normalized
    by each candidate's rollout count, plus its footprint."""
    k = len(obs)
    n = np.maximum(obs["n"].astype(np.float32), 1.0)[:, None, None]
    planes = np.zeros((k, 5, BOARD, BOARD), dtype=np.float32)
    for j, name in enumerate(_COUNT_PLANES):
        planes[:, j] = obs[name].astype(np.float32).reshape(k, BOARD, BOARD) / n
    for i in range(k):
        planes[i, 4] = move_footprint(moves[i])
    return planes


def _observed_scalars(obs: np.ndarray) -> np.ndarray:
    """(K,) .sobs records -> (K, 6) [win/draw/loss freq, delta mean/std (score
    points, ~unit-scaled), log1p rollouts]."""
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


def _predicted_scalars(first_pass: dict[str, torch.Tensor]) -> np.ndarray:
    """First-pass outputs for the K evidence candidates -> (K, 5)
    [p_win, p_draw, p_loss, sd mean, sd std], the sd pair unit-scaled the way
    the observed delta moments are."""
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
    arrays, or any prefix of them); `pre_move_diff` is the mover's score
    differential before the move, as the move encoder expects; `first_pass`
    holds the model's evidence-free outputs ("wld", "score_diff", "planes")
    sliced to the same K candidates in the same order.
    """
    k = len(moves)
    if k > max_e:
        raise ValueError(f"evidence set of {k} candidates does not fit max_e={max_e}")
    if first_pass["planes"].shape[0] != k:
        raise ValueError("first-pass rows do not match the evidence candidates")
    if k == 0:
        return empty_evidence_inputs(max_e, dtype=dtype, device=device)

    enc = encode_moves(np.asarray(moves), np.full(k, pre_move_diff, dtype=np.int32))

    observed = _observed_planes(moves, obs)
    predicted = torch.sigmoid(first_pass["planes"].detach()).cpu().float().numpy()
    planes = np.zeros((k, NUM_EVIDENCE_PLANES, BOARD, BOARD), dtype=np.float32)
    planes[:, :4] = observed[:, :4]
    planes[:, 4:8] = predicted.reshape(k, 4, BOARD, BOARD)
    planes[:, 8] = observed[:, 4]

    scalars = np.concatenate([_observed_scalars(obs), _predicted_scalars(first_pass)], axis=1)
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
    """The empty evidence set as (1, max_e, ...) inputs -- every row masked
    out, so the model degrades to its plain pass (the prefix-0 training rows,
    and the first pass of every deployed turn). Row widths follow the move
    encoding the way build_evidence_inputs' do."""
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
