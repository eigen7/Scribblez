"""Whether a transfer_test corpus carries transfer signal at all, read without
a model (docs/plans/supreme_bot_m1a.md, "Is there signal to transfer?").

A candidate's held-out value can only be learned from other candidates'
probes if those probes say something about it. One thing they can say: probe
i of every candidate deals the opponent the same rack, so a reply that hurt
candidate a would also be open after candidate b, unless b blocks it (engine
sim/reply_blocking.h). The damage b blocks is then the mean, over the other
candidates' probes whose replies b blocks, of how much worse that rollout went
than its candidate's mean, averaged over all the other candidates' probes.

The test: within each position (centered over its candidates), does the
damage a candidate blocks predict its label minus the teacher prior, the part
of its value the prior misses? A positive correlation is signal a reader
could transfer; the control pairs each candidate with another position's
damage blocked.

Positions whose plausible candidates' labels are all equal are decided (as in
evaluate.py) and left out, as are positions with fewer than three
candidates. "Threat" positions are the THREAT_SHARE with the largest
within-row spread of damage blocked: where blocking matters most.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scribblez.ffi import reply_blocking
from scribblez.transfer_test.corpus import CorpusFile
from scribblez.transfer_test.evaluate import BAGS, decided

MIN_CANDIDATES = 3
THREAT_SHARE = 0.03
CONTROL_SEED = 0


@dataclass
class Candidates:
    """Every scored candidate of a corpus, position by position."""

    group: np.ndarray  # (N,) the candidate's position, 0-based over the corpus
    bag: np.ndarray  # (N,) tiles in the bag at its position
    plausible: np.ndarray  # (N,) bool: top or middle stratum
    label: np.ndarray  # (N,) expected score (win + draw / 2)
    label_var: np.ndarray  # (N,) its sampling variance
    prior: np.ndarray  # (N,) the teacher's expected score
    damage_blocked: np.ndarray  # (N,)
    share_blocked: np.ndarray  # (N,) share of the other candidates' replies it blocks


def blocking_columns(blocked: np.ndarray, damage: np.ndarray, probes: int) -> np.ndarray:
    """(2, K) damage blocked and share blocked per candidate, from a position's
    (K * probes, K) blocking matrix and each record's damage."""
    k = blocked.shape[1]
    others = np.repeat(np.arange(k), probes)[:, None] != np.arange(k)[None, :]
    b = blocked.astype(np.float64) * others
    n = others.sum(axis=0)
    return np.stack([(b * damage[:, None]).sum(axis=0) / n, b.sum(axis=0) / n])


def position_columns(f: CorpusFile, blocked: np.ndarray, p: int) -> dict[str, np.ndarray] | None:
    """Position p's candidates' columns of Candidates (without group), or None
    when it is decided or too small."""
    pr = f.probes
    c0, c1 = int(pr.candidate_start[p]), int(pr.candidate_start[p + 1])
    k = c1 - c0
    lab = f.labels[c0:c1]
    n = lab["n"].astype(np.float64)
    label = (lab["wins"] + 0.5 * lab["draws"]) / n
    plausible = pr.candidates["stratum"][c0:c1] <= 1
    if k < MIN_CANDIDATES or decided(label, plausible):
        return None
    rec = pr.records[c0 * pr.probes : c1 * pr.probes]
    outcome = (rec["p_win"] + 0.5 * rec["p_draw"]).astype(np.float64).reshape(k, pr.probes)
    damage = (outcome.mean(axis=1, keepdims=True) - outcome).reshape(-1)  # > 0: went worse
    damage_blocked, share_blocked = blocking_columns(
        blocked[c0 * pr.probes : c1 * pr.probes, :k], damage, pr.probes
    )
    wld = f.prior.wld[c0:c1]
    return {
        "bag": np.full(k, int(f.replay.roots["bag_size"][p])),
        "plausible": plausible,
        "label": label,
        "label_var": label * (1 - label) / n,
        "prior": (wld[:, 0] + 0.5 * wld[:, 1]).astype(np.float64),
        "damage_blocked": damage_blocked,
        "share_blocked": share_blocked,
    }


def file_blocking(f: CorpusFile, threads: int) -> np.ndarray:
    """(records, max candidates) blocking matrix of a corpus file."""
    pr = f.probes
    stride = int(np.diff(pr.candidate_start).max())
    return reply_blocking(pr.path.with_suffix(".slog"), pr.path, len(pr.records), stride, threads)


def collect(files: list[CorpusFile], threads: int, progress=None) -> Candidates:
    """The scored candidates of every file. `progress(done, total)` is called
    after each file."""
    columns: list[dict[str, np.ndarray]] = []
    for i, f in enumerate(files):
        blocked = file_blocking(f, threads)
        for p in range(f.num_positions):
            cols = position_columns(f, blocked, p)
            if cols is not None:
                cols["group"] = np.full(len(cols["label"]), len(columns))
                columns.append(cols)
        if progress:
            progress(i + 1, len(files))
    return Candidates(**{name: np.concatenate([c[name] for c in columns]) for name in columns[0]})


def centered(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """x minus its position's mean."""
    return x - (np.bincount(group, weights=x) / np.bincount(group))[group]


def within_row_spread(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Per position, the standard deviation of a centered column."""
    return np.sqrt(np.bincount(group, weights=x * x) / np.bincount(group))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def slice_stats(c: Candidates, mask: np.ndarray) -> dict[str, float]:
    """On the masked candidates, centered: how the damage blocked relates to
    the prior's miss (label minus prior), to the prior and to the label."""
    damage = centered(c.damage_blocked, c.group)[mask]
    miss = centered(c.label - c.prior, c.group)[mask]
    r = correlation(damage, miss)
    slope = float(np.polyfit(damage, miss, 1)[0])
    return {
        "candidates": int(mask.sum()),
        "positions": len(np.unique(c.group[mask])),
        "corr_damage_miss": r,
        "r2_damage_miss": r * r,
        "slope_damage_miss": slope,
        "corr_share_miss": correlation(centered(c.share_blocked, c.group)[mask], miss),
        "corr_damage_prior": correlation(damage, centered(c.prior, c.group)[mask]),
        "corr_damage_label": correlation(damage, centered(c.label, c.group)[mask]),
        "miss_sd": float(miss.std()),
        "residual_sd": float((miss - slope * damage).std()),
    }


def shuffled_control(c: Candidates, seed: int) -> float:
    """The correlation with each candidate's damage blocked taken from a
    random other candidate of the corpus."""
    damage = centered(c.damage_blocked, c.group)
    miss = centered(c.label - c.prior, c.group)
    return correlation(np.random.default_rng(seed).permutation(damage), miss)


def threat_mask(c: Candidates) -> np.ndarray:
    """The candidates of the THREAT_SHARE positions with the widest spread of
    damage blocked."""
    spread = within_row_spread(centered(c.damage_blocked, c.group), c.group)
    cut = np.quantile(spread, 1 - THREAT_SHARE)
    return spread[c.group] >= cut


def report(c: Candidates) -> dict:
    slices = {"all": np.ones(len(c.group), dtype=bool)}
    for lo, hi in BAGS:
        slices[f"bag {lo}-{hi}"] = (c.bag >= lo) & (c.bag <= hi)
    slices[f"threat top {THREAT_SHARE:.0%}"] = threat_mask(c)
    return {
        "slices": {name: slice_stats(c, m) for name, m in slices.items()},
        "shuffled_control_corr": shuffled_control(c, CONTROL_SEED),
        "label_noise_sd": float(np.sqrt(c.label_var.mean())),
    }
