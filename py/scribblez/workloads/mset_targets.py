"""The teacher-labeling step shared by move_set_eval and evidence_trajectories:
running move_set_eval_target_generator over .slog files to give them .mset
sidecars, plus helpers for the model files those workloads name.

The generator has two candidate-selection modes with disjoint flags, so one run
uses one mode:

  - stratified: a small sample per position across the equity ranking.
    evidence_trajectories adds --sobs to force-include its simmed candidates.
  - full sweep: every legal candidate of a few positions, for move_set_eval's
    held-out slice.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scribblez.paths import ENGINE_DIR

TARGET_GENERATOR = str(ENGINE_DIR / "move_set_eval_target_generator")


@dataclass(frozen=True)
class StratifiedQuotas:
    """The stratified candidate sample per position: the head of the equity
    ranking, a sample of the contention zone (ranks up to mid_rank_limit), a
    uniform sample of the remaining ranks, and exchanges."""

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
    positions per game, capped by static-equity rank."""
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


def pin_model(path: str, paths, name: str) -> Path:
    """The tag's own copy of the model export at `path`, made under the tag
    root's pinned/ on first use and reused afterwards.

    A param naming another tag's export cannot read it in place for the life of
    this tag: a move_set_eval tag prunes its exports as it trains
    (move_set_eval.trainer.prune_exports). Raises FileNotFoundError when neither
    the copy nor the source exists."""
    if not path:
        raise FileNotFoundError(f"{name} is unset")
    dest = Path(paths.root) / "pinned" / Path(path).name
    if dest.is_file():
        return dest
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name} {path!r} is not a readable file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Copy to a per-process temp file and rename it over the destination, so a
    # reader never sees a partial copy. Concurrent first users (a tag's workers
    # starting together, or a worker and the dashboard) each land a whole file;
    # the last rename wins, with identical bytes.
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    shutil.copyfile(path, tmp)
    os.replace(tmp, dest)
    return dest


def require_model_file(path: str, name: str) -> bool:
    """Fail fast on a model path the whole run would trip over."""
    if path and Path(path).is_file():
        return True
    print(f"error: {name} {path!r} is not a readable file", file=sys.stderr)
    return False
