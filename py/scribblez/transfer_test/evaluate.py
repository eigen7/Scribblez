"""M1a's evaluation harness, first part (docs/plans/supreme_bot_m1a.md, PR 5):
a reader and the baseline arms scored on the test corpus.

Rows. Each test position is assembled `replicates` times under a fixed seed,
with the reader's training recipe but queried at the full context only, so
every arm sees the same held-out moves and the same kept probes.

Arms, each an expected score (win + draw / 2) per candidate:

    prior          the teacher's, or a constant under the run's prior "none"
    shrinkage      each probed candidate's mean probe outcome shrunk toward its
                   prior with the weight of `k` probes; held-out candidates keep
                   the prior. `k` is fitted on the test rows' probed candidates,
                   which the headline does not score, and so favours the
                   baselines rather than the reader
    common shift   shrinkage, with each held-out candidate also moved by the
                   mean shrinkage correction of its row's probed candidates
    reader         the reader at the full context
    shuffled       the reader with each row's probe outcomes permuted among its
                   probes: evidence of the same shape that no longer belongs to
                   its candidates (a control)
    no_probes      the reader with every probe removed from the context: what
                   it knows from the candidates and the board alone (a control)

Metrics, over positions whose plausible candidates' labels differ (decided
positions are counted, not scored), overall and by tiles in the bag:

    held-out error   the within-row centered RMSE of the expected score on the
                     held-out candidates: M1a's headline
    probed error     the same on the probed candidates
    pair accuracy    among pairs of held-out candidates in a row whose labels
                     differ by more than twice their combined noise, the share
                     the arm orders correctly (ties count half)

Each arm's difference from the prior gets a 95% interval by bootstrap over
positions, and so does each from the shuffled control. Without the teacher's
prior the reader can learn a static evaluator of candidates, which beats a
constant prior with no transfer at all, so there the headline is the
reader's held-out error minus the shuffled control's: what the evidence
itself adds.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from scribblez import params as params_mod
from scribblez.transfer_test.corpus import CorpusFile
from scribblez.transfer_test.reader import Reader, ReaderConfig
from scribblez.transfer_test.rows import LEAF, ROOT_TOKENS, Row, RowConfig, assemble_row
from scribblez.transfer_test.tokens import collate
from scribblez.workloads.transfer_reader import TransferReaderParams

ARMS = ("prior", "shrinkage", "common_shift", "reader", "shuffled", "no_probes")
READER_ARMS = ("reader", "shuffled", "no_probes")
BAGS = ((1, 3), (4, 7), (8, 11), (12, 15))
SHRINK_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PAIR_NOISE_MULTIPLE = 2.0
EVAL_SEED = 2026
BOOTSTRAP = 1000


def load_reader(path, device) -> tuple[Reader, dict]:
    """The reader in a trainer checkpoint, in eval mode on `device`, and the
    checkpoint's config (the run's params plus the reader's shape)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    reader = Reader(ReaderConfig(**ckpt["config"]["reader"]))
    reader.load_state_dict(ckpt["model_state_dict"])
    return reader.to(device).eval(), ckpt["config"]


def run_params(config: dict) -> TransferReaderParams:
    """The run's params from a checkpoint's config, defaults filled in."""
    return params_mod.validate(
        TransferReaderParams, {k: v for k, v in config.items() if k != "reader"}
    )


def eval_row_config(params: dict) -> RowConfig:
    """The run's row recipe, queried at the full context only."""
    return RowConfig(
        max_held_out=params["max_held_out"],
        max_probes=params["max_probes"],
        max_tokens=params["max_tokens"],
        query_points=1,
        graded_max=params["graded_max"],
    )


@dataclass
class Scored:
    """One assembled row's candidates: labels, the evidence the arms read,
    and each arm's expected score."""

    file: int
    position: int
    bag: int
    label: np.ndarray  # (K,) expected score
    label_var: np.ndarray  # (K,) its sampling variance bound
    prior: np.ndarray  # (K,)
    probe_sum: np.ndarray  # (K,) kept probes' expected scores, summed
    probe_count: np.ndarray  # (K,)
    held: np.ndarray  # (K,) bool
    plausible: np.ndarray  # (K,) bool: top or middle stratum
    arms: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.plausible.sum() < 2 or np.ptp(self.label[self.plausible]) < 1e-9


def probe_evidence(row: Row, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each candidate's kept probes: summed expected score and count."""
    leaves = row.kind == LEAF
    leaf_slots, outcomes = row.slot[leaves], row.leaf[row.ref[leaves]]
    outcome = outcomes[:, 0] + 0.5 * outcomes[:, 1]
    return (
        np.bincount(leaf_slots, weights=outcome, minlength=k),
        np.bincount(leaf_slots, minlength=k).astype(np.float64),
    )


def score_row(f: CorpusFile, file_index: int, row: Row) -> Scored:
    t, pv = row.target, row.candidate["prior_value"]
    label = t["wld"][:, 0] + 0.5 * t["wld"][:, 1]
    k = len(label)
    probe_sum, probe_count = probe_evidence(row, k)
    return Scored(
        file=file_index,
        position=row.position,
        bag=int(f.replay.roots["bag_size"][row.position]),
        label=label.astype(np.float64),
        label_var=np.maximum(label * (1 - label), 1e-4) / t["n"],
        prior=(pv[:, 0] + 0.5 * pv[:, 1]).astype(np.float64),
        probe_sum=probe_sum,
        probe_count=probe_count,
        held=row.held_out.copy(),
        plausible=row.candidate["stratum"] <= 1,
    )


def shrunk(s: Scored, k: float) -> np.ndarray:
    return (s.probe_sum + k * s.prior) / (s.probe_count + k)


def centered_sq_errors(s: Scored, estimate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Squared within-row errors on `mask`, centering prediction and label
    over the row's plausible candidates."""
    p = s.plausible
    err = (estimate - estimate[p].mean()) - (s.label - s.label[p].mean())
    return err[mask & p] ** 2


def fit_shrinkage(scored: list[Scored]) -> float:
    """The SHRINK_GRID weight with the lowest probed error."""

    def probed_error(k: float) -> float:
        sq = [centered_sq_errors(s, shrunk(s, k), ~s.held) for s in scored if not s.decided]
        return float(np.mean(np.concatenate(sq)))

    return min(SHRINK_GRID, key=probed_error)


def baseline_arms(s: Scored, k: float):
    s.arms["prior"] = s.prior
    estimate = np.where(s.held, s.prior, shrunk(s, k))
    s.arms["shrinkage"] = estimate
    probed = ~s.held & (s.probe_count > 0)
    shift = (estimate - s.prior)[probed].mean() if probed.any() else 0.0
    s.arms["common_shift"] = np.where(s.held, s.prior + shift, estimate)


def shuffled(row: Row, rng: np.random.Generator) -> Row:
    """`row` with its probe outcomes permuted among its probes."""
    return dataclasses.replace(row, leaf=row.leaf[rng.permutation(len(row.leaf))])


def without_probes(row: Row) -> Row:
    """`row` with its context cut to the root and candidate tokens, every
    query asked there. The probes' feature tables stay, unreferenced."""
    n = ROOT_TOKENS + len(row.held_out)
    return dataclasses.replace(
        row,
        kind=row.kind[:n],
        slot=row.slot[:n],
        ply=row.ply[:n],
        ref=row.ref[:n],
        query_prefix=np.full_like(row.query_prefix, n),
    )


@torch.no_grad()
def reader_estimates(reader: Reader, rows: list[Row], device, batch: int = 32) -> list[np.ndarray]:
    """Each row's per-candidate expected score from the reader at the full
    context (one query per candidate, in slot order)."""
    q_len = reader.cfg.max_slots
    max_tokens = max(len(r.kind) for r in rows)
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        b = collate(chunk, max_tokens, q_len).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reader(b)["expected"][..., 0].float().cpu().numpy()
        for j, r in enumerate(chunk):
            out.append(expected[j, : len(r.held_out)].astype(np.float64))
    return out


def assemble(files: list[CorpusFile], cfg: RowConfig, replicates: int, seed: int):
    """(each reader arm's rows, by READER_ARMS name; their scored records)."""
    rng = np.random.default_rng(seed)
    rows, scored = {arm: [] for arm in READER_ARMS}, []
    for i, f in enumerate(files):
        for p in range(f.num_positions):
            for _ in range(replicates):
                row = assemble_row(f, i, p, cfg, rng)
                rows["reader"].append(row)
                rows["shuffled"].append(shuffled(row, rng))
                rows["no_probes"].append(without_probes(row))
                scored.append(score_row(f, i, row))
    return rows, scored


def _per_arm() -> dict[str, float]:
    return defaultdict(float)


@dataclass
class PositionTotals:
    """One position's sums over its replicates, per arm."""

    bag: int
    held_sq: dict[str, float] = dataclasses.field(default_factory=_per_arm)
    held_n: int = 0
    probed_sq: dict[str, float] = dataclasses.field(default_factory=_per_arm)
    probed_n: int = 0
    pairs_correct: dict[str, float] = dataclasses.field(default_factory=_per_arm)
    pairs: int = 0


def resolved_pairs(s: Scored) -> list[tuple[int, int]]:
    """Pairs of held-out plausible candidates whose labels differ by more than
    PAIR_NOISE_MULTIPLE times their combined noise."""
    held = np.flatnonzero(s.held & s.plausible)
    return [
        (a, b)
        for a, b in itertools.combinations(held, 2)
        if abs(s.label[a] - s.label[b])
        > PAIR_NOISE_MULTIPLE * np.sqrt(s.label_var[a] + s.label_var[b])
    ]


def position_totals(scored: list[Scored]) -> list[PositionTotals]:
    by_position = defaultdict(list)
    for s in scored:
        if not s.decided:
            by_position[(s.file, s.position)].append(s)
    totals = []
    for group in by_position.values():
        t = PositionTotals(group[0].bag)
        for s in group:
            pairs = resolved_pairs(s)
            t.held_n += int((s.held & s.plausible).sum())
            t.probed_n += int((~s.held & s.plausible).sum())
            t.pairs += len(pairs)
            for arm, est in s.arms.items():
                t.held_sq[arm] += centered_sq_errors(s, est, s.held).sum()
                t.probed_sq[arm] += centered_sq_errors(s, est, ~s.held).sum()
                for a, b in pairs:
                    order = np.sign(est[a] - est[b]) * np.sign(s.label[a] - s.label[b])
                    t.pairs_correct[arm] += 0.5 * (order + 1)
        totals.append(t)
    return totals


def metrics(totals: list[PositionTotals]) -> dict[str, dict[str, float]]:
    """Per arm: held-out and probed error, pair accuracy."""
    held_n = sum(t.held_n for t in totals)
    probed_n = sum(t.probed_n for t in totals)
    pairs = sum(t.pairs for t in totals)
    return {
        arm: {
            "heldout_rmse": float(np.sqrt(sum(t.held_sq[arm] for t in totals) / max(held_n, 1))),
            "probed_rmse": float(np.sqrt(sum(t.probed_sq[arm] for t in totals) / max(probed_n, 1))),
            "pair_accuracy": float(sum(t.pairs_correct[arm] for t in totals) / max(pairs, 1)),
        }
        for arm in ARMS
    }


def bootstrap_intervals(
    totals: list[PositionTotals], seed: int, baseline: str
) -> dict[str, dict[str, list]]:
    """95% intervals of each arm's metrics minus the `baseline` arm's,
    resampling positions."""
    rng = np.random.default_rng(seed)
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(BOOTSTRAP):
        sample = [totals[i] for i in rng.integers(len(totals), size=len(totals))]
        m = metrics(sample)
        for arm in ARMS:
            for key, value in m[arm].items():
                draws[arm][key].append(value - m[baseline][key])
    return {
        arm: {key: np.percentile(v, [2.5, 97.5]).tolist() for key, v in by_key.items()}
        for arm, by_key in draws.items()
    }


def report(totals: list[PositionTotals], seed: int) -> dict:
    """The whole report: overall and per bag band, with intervals overall."""
    out = {
        "positions": len(totals),
        "heldout_candidates": sum(t.held_n for t in totals),
        "pairs": sum(t.pairs for t in totals),
        "overall": metrics(totals),
        "intervals_vs_prior": bootstrap_intervals(totals, seed, "prior"),
        "intervals_vs_shuffled": bootstrap_intervals(totals, seed, "shuffled"),
        "by_bag": {},
    }
    for lo, hi in BAGS:
        band = [t for t in totals if lo <= t.bag <= hi]
        if band:
            out["by_bag"][f"{lo}-{hi}"] = {"positions": len(band), **metrics(band)}
    return out


def evaluate(
    reader: Reader,
    params: dict,
    files: list[CorpusFile],
    device,
    replicates: int = 4,
    seed: int = EVAL_SEED,
) -> dict:
    """Score `reader` and the baseline arms on `files` (see the module
    docstring)."""
    rows, scored = assemble(files, eval_row_config(params), replicates, seed)
    k = fit_shrinkage(scored)
    for s in scored:
        baseline_arms(s, k)
    for arm, arm_rows in rows.items():
        for s, est in zip(scored, reader_estimates(reader, arm_rows, device), strict=True):
            s.arms[arm] = est
    totals = position_totals(scored)
    decided = len({(s.file, s.position) for s in scored if s.decided})
    return {"shrinkage_probes": k, "decided_positions": decided, **report(totals, seed)}
