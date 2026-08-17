"""The teacher-labeling step shared by the pair-producing workloads: running
move_set_eval_target_generator over .slog files to give them .mset sidecars.

move_set_eval labels its pairs stratified (plus a full-sweep held-out slice);
evidence_trajectories labels stratified with the simmed trajectory candidates
force-included. The generator's two candidate-selection modes take disjoint
parameters, so a run is one or the other.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scribblez.paths import ENGINE_DIR

TARGET_GENERATOR = str(ENGINE_DIR / "move_set_eval_target_generator")


@dataclass(frozen=True)
class StratifiedQuotas:
    """The stratified candidate sample per position: dense head of the equity
    ranking, a slice of the contention zone, a uniform tail, and exchanges."""

    top: int
    mid: int
    tail: int
    exchange: int
    mid_rank_limit: int  # exclusive rank bound of the contention zone

    @classmethod
    def from_params(cls, params) -> "StratifiedQuotas":
        """From a params dataclass carrying the quota_* / mid_rank_limit fields."""
        return cls(
            params.quota_top,
            params.quota_mid,
            params.quota_tail,
            params.quota_exchange,
            params.mid_rank_limit,
        )


def label_stratified(
    pending: list[Path],
    teacher_model: str,
    quotas: StratifiedQuotas,
    positions_per_game: int,
    threads: int,
    with_sobs: bool = False,
) -> int:
    """Label `pending` .slog files with the stratified sample. `with_sobs`
    force-includes each position's simmed trajectory candidates from the
    same-stem .sobs sidecar."""
    selection = [
        f"--quota-top={quotas.top}",
        f"--quota-mid={quotas.mid}",
        f"--quota-tail={quotas.tail}",
        f"--quota-exchange={quotas.exchange}",
        f"--mid-rank-limit={quotas.mid_rank_limit}",
        f"--positions-per-game={positions_per_game}",
        *(["--sobs"] if with_sobs else []),
    ]
    return _run(pending, teacher_model, selection, threads)


def label_full_sweep(
    pending: list[Path],
    teacher_model: str,
    candidate_cap: int,
    positions_per_game: int,
    threads: int,
) -> int:
    """Label `pending` .slog files with every legal candidate of a few
    positions per game (capped by static-equity rank)."""
    selection = [
        "--full-sweep",
        f"--sweep-cap={candidate_cap}",
        f"--positions-per-game={positions_per_game}",
    ]
    return _run(pending, teacher_model, selection, threads)


def _run(pending: list[Path], teacher_model: str, selection: list[str], threads: int) -> int:
    cmd = [
        TARGET_GENERATOR,
        *[f"--slog-file={p}" for p in pending],
        f"--model={teacher_model}",
        *selection,
        f"--threads={threads}",
    ]
    rc = subprocess.run(cmd, capture_output=False).returncode
    if rc != 0:
        print(f"move_set_eval_target_generator exited with code {rc}", file=sys.stderr)
    return rc


def require_model_file(path: str, name: str) -> bool:
    """Fail fast on a model path the whole run would trip over."""
    if path and Path(path).is_file():
        return True
    print(f"error: {name} {path!r} is not a readable file", file=sys.stderr)
    return False
