"""M1a's corpus in memory: per .sprobe file, its probes, their replayed state,
the labels and the prior cache, checked against one another as they load.

Labels keep only what the reader trains on (outcome counts, score moments,
and the opponent's-reply and own-next footprint histograms), about a third of
a .sobs record; the win-weighted histograms are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.transfer_test.prior import Prior, prior_path, read_prior
from scribblez.transfer_test.probes import (
    ProbeFile,
    ProbeReplay,
    read_labels,
    read_sprobe,
    replay,
    tile_codes,
    tile_counts,
)
from scribblez.workloads.pair_store import complete_pairs
from scribblez.workloads.transfer_test import PROBE_EXT

LABEL_FIELDS = (
    "wins",
    "draws",
    "losses",
    "delta_sum",
    "delta_sq_sum",
    "n",
    "opp_next_count",
    "self_next_count",
)


@dataclass
class CorpusFile:
    probes: ProbeFile
    replay: ProbeReplay
    labels: np.ndarray  # (C,) LABEL_FIELDS of the label records
    prior: Prior
    drew: np.ndarray  # (T,) bool: the turn drew tiles

    @property
    def num_positions(self) -> int:
        return self.probes.num_positions


def slim_labels(obs: np.ndarray) -> np.ndarray:
    dtype = np.dtype([(name, obs.dtype.fields[name][0]) for name in LABEL_FIELDS])
    out = np.empty(len(obs), dtype=dtype)
    for name in LABEL_FIELDS:
        out[name] = obs[name]
    return out


def load_file(path: Path) -> CorpusFile:
    """Load one .sprobe with its .slog replay, labels and prior. Raises if any
    part is missing or does not match the probes."""
    probes = read_sprobe(path)
    prior = read_prior(prior_path(path))
    if len(prior.root_board) != probes.num_positions or len(prior.wld) != len(probes.candidates):
        raise ValueError(f"{prior_path(path)} does not match {path}")
    state = replay(probes)
    return CorpusFile(
        probes=probes,
        replay=state,
        labels=slim_labels(read_labels(probes)),
        prior=prior,
        drew=tile_counts(tile_codes(state.turns["drawn"])).sum(axis=1) > 0,
    )


def load_corpus(store_dir: Path) -> list[CorpusFile]:
    """Every complete file in a transfer_test tag's corpus store, in name
    order."""
    return [load_file(p) for p in complete_pairs(store_dir, PROBE_EXT)]
