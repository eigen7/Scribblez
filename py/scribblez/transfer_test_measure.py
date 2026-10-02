"""Step 0 of SupremeBot M1a (docs/plans/supreme_bot_m1a.md): reading the
transfer_test workload's measurement sidecars.

The question is how many rollouts L a label needs before within-position
differences between candidates are resolvable. M1a's headline compares a
held-out move with its siblings, so the label quantity that matters is a
candidate's expected score centered on its position's candidates. Centering
happens per rollout: under common random numbers rollout i deals every
candidate the same opponent rack, so subtracting the rollout's mean over the
candidates removes most of that rollout's luck.

For one position, with d[c, i] the centered expected score of candidate c in
rollout i, the label at budget n is the mean of d[c, :n]. Its noise variance is
var_i(d[c]) / n, averaged over the candidates. Its signal variance is the
spread of the labels across the candidates at the full budget, less the noise
still in them: centered over K candidates, the labels' spread carries K/(K-1)
times the per-candidate noise var_i(d[c]) / n. A budget resolves a position
when noise / signal falls below a target ratio.

Exchanges and low-ranked plays sit far below the best plays, so a spread over
every candidate is dominated by them. The comparisons that are hard to
resolve are among the plausible moves, so the report also measures the top and
middle strata alone, centered over those candidates only.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JSON_EXT = ".tmeasure"
FLOATS_EXT = ".trollouts"


@dataclass(frozen=True)
class MeasuredPosition:
    game: int
    turn: int
    strata: list[str]  # per candidate
    expected: np.ndarray  # (K, R) expected score W + D/2, root mover's POV
    delta: np.ndarray  # (K, R) final score difference
    couplings: list[dict]  # the coupled pairs among the candidates
    offered: dict[str, int]  # the coupled pairs the position offered, by kind
    saturation: list[list[list[int]]]  # per candidate: [[probes, options], ...]


@dataclass(frozen=True)
class MeasuredFile:
    header: dict
    positions: list[MeasuredPosition]


def read_file(json_path: Path) -> MeasuredFile:
    doc = json.loads(json_path.read_text())
    floats = np.fromfile(json_path.with_suffix(FLOATS_EXT), "<f4")
    rollouts = doc["rollouts"]
    positions = []
    for p in doc["positions"]:
        k = len(p["candidates"])
        block = floats[p["float_offset"] : p["float_offset"] + 2 * k * rollouts]
        pairs = block.reshape(k, rollouts, 2)
        positions.append(
            MeasuredPosition(
                game=p["game"],
                turn=p["turn"],
                strata=[c["stratum"] for c in p["candidates"]],
                expected=pairs[:, :, 0].astype(np.float64),
                delta=pairs[:, :, 1].astype(np.float64),
                couplings=p["couplings"],
                offered=p["offered_couplings"],
                saturation=p["saturation"],
            )
        )
    return MeasuredFile(
        header={k: v for k, v in doc.items() if k != "positions"}, positions=positions
    )


def read_store(store: Path) -> list[MeasuredFile]:
    return [read_file(p) for p in sorted(store.glob(f"*{JSON_EXT}"))]


def rows_in(position: MeasuredPosition, values: np.ndarray, strata: set[str]) -> np.ndarray:
    """The (K', R) rows of `values` whose candidates are in `strata`."""
    return values[[s in strata for s in position.strata]]


def centered(values: np.ndarray) -> np.ndarray:
    """(K, R) values minus each rollout's mean over the candidates."""
    return values - values.mean(axis=0, keepdims=True)


@dataclass(frozen=True)
class NoiseSignal:
    noise_per_rollout: float  # mean over candidates of var_i(d[c]): noise variance at n = 1
    signal: float  # variance across candidates of the full-budget labels, noise removed


def noise_signal(values: np.ndarray) -> NoiseSignal:
    """The noise and signal variances of one position's centered labels."""
    d = centered(values)
    k, r = d.shape
    per_rollout = float(d.var(axis=1, ddof=1).mean())
    labels = d.mean(axis=1)
    signal = float(labels.var(ddof=1)) - per_rollout / r * k / (k - 1)
    return NoiseSignal(per_rollout, signal)


def noise_to_signal(ns: NoiseSignal, n: int) -> float:
    """noise / signal at budget n; inf when the position has no resolvable signal."""
    return ns.noise_per_rollout / n / ns.signal if ns.signal > 0 else float("inf")


def independent_to_paired_variance(values: np.ndarray) -> float:
    """How much centering per rollout shrinks the per-rollout variance: the mean
    variance of the raw values over that of the centered ones."""
    return float(values.var(axis=1, ddof=1).mean() / centered(values).var(axis=1, ddof=1).mean())


def rollouts_per_second(files: list[MeasuredFile]) -> float:
    """Rollouts simmed per wall-clock second, over every file."""
    total = sum(sum(p.expected.size for p in f.positions) for f in files)
    seconds = sum(f.header["sim_seconds"] for f in files)
    return total / seconds if seconds > 0 else float("nan")


def saturation_by_stratum(files: list[MeasuredFile]) -> dict[str, dict[int, list[int]]]:
    """Options recorded per candidate board, by stratum and probe count."""
    out: dict[str, dict[int, list[int]]] = {}
    for f in files:
        for p in f.positions:
            for stratum, curve in zip(p.strata, p.saturation, strict=True):
                by_n = out.setdefault(stratum, {})
                for probes, options in curve:
                    by_n.setdefault(probes, []).append(options)
    return out
