"""Workload: collect HastyBot's blind spots -- positions where a play from
outside its static-equity top moves out-sims all of them.

One role, a generator that any number of workers run side by side. A cycle
plays one HastyBot-vs-HastyBot game and surveys every eligible turn of it
(sim_candidate_survey_tool: a racing screen of every legal play, then a longer
confirming sim, with solved endgames late in the game, of the top moves and
the screen's best plays from outside them). Games cost nothing beside the
sims, so a game a cycle wastes none, and workers need no coordination: each
plays its own randomly seeded games. The controller ends the run: its
scheduler tick parks every surveyor once the tag holds target_positions, and
the dashboard's idle policy then stops the rented machines.

What a cycle delivers is small. A game's full survey file runs to megabytes
and nearly all of it describes positions where the top moves were fine, so
the worker keeps only the positions it found (slim_survey) and their .gcg
exports:

    data/survey/<stem>.simsurvey.json   the found positions of one game
    data/gcg/<stem>-g0-turn<N>.gcg      the game up to each found position

`<stem>` carries the worker id, so names are unique across workers. The two
directories together are what py/scripts/sim_survey_viewer.py browses and
what py/scripts/blind_spots_collect.py turns into a committed examples
directory.
"""

import hashlib
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from scribblez import params as params_mod
from scribblez.params import param
from scribblez.paths import ENGINE_DIR
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.sim_candidate_survey import SURVEY_SUFFIX, gcg_name, slim_survey_file
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.worker import WorkerStats, WorkerStopped

SURVEY_TOOL = str(ENGINE_DIR / "sim_candidate_survey_tool")

# The tag's data/ subdirectories (shared by delivery, sync and progress).
SURVEY_DIR = "survey"
GCG_DIR = "gcg"


@dataclass(frozen=True)
class BlindSpotsParams:
    target_positions: int = param(
        100, "stop every worker once the tag holds this many found positions (0: never)"
    )
    face_up_leaves: bool = param(True, "play and sim with the opponent's kept tiles known")
    cut: int = param(10, "HastyBot's top-K moves by static equity that a play must beat")
    rollouts: int = param(
        1000, "screening rollouts per legal play (clearly beaten plays stop early)"
    )
    confirm_rollouts: int = param(5000, "confirming rollouts per top move and per outside pick")
    confirm_picks: int = param(5, "plays from outside the top moves that get a confirming sim")
    solve_max_unseen: int = param(
        14,
        "confirming rollouts solve endgames at or below this many unseen tiles (-1: never)",
    )
    random_opening_mean: float = param(2.0, "mean random opening plies per game")


def play_game(work_dir: Path, worker_id: str, params: BlindSpotsParams) -> Path:
    """Play one game into `work_dir`, named <timestamp>-<worker_id>.slog so every
    file the survey derives from it is unique across workers."""
    before = set(work_dir.glob("*.slog"))
    code = run_games(
        work_dir,
        num_games=1,
        threads=1,
        player_spec=hasty_player_spec(),
        random_opening_mean=params.random_opening_mean,
        face_up_leaves=params.face_up_leaves,
    )
    if code != 0:
        raise RuntimeError(f"play_game exited {code}")
    (fresh,) = set(work_dir.glob("*.slog")) - before
    return fresh.rename(fresh.with_name(f"{fresh.stem}-{worker_id}.slog"))


def survey_seed(work_dir: Path) -> int:
    """The survey seed for the game in `work_dir`, derived from its .slog name.
    The tool resumes a partial survey only under the header it started with,
    seed included, so a run restarted mid-game must pass the same seed."""
    names = "\n".join(sorted(p.name for p in work_dir.glob("*.slog")))
    return int.from_bytes(hashlib.blake2b(names.encode(), digest_size=8).digest()) >> 2


def run_survey_tool(work_dir: Path, params: BlindSpotsParams, threads: int) -> int:
    """Survey every eligible turn of each game in `work_dir` that has no finished
    survey file yet (a game interrupted mid-survey resumes from its partial
    file)."""
    # fmt: off
    cmd = [
        SURVEY_TOOL,
        "--slog-dir", str(work_dir),
        "--gcg-dir", str(work_dir / GCG_DIR),
        "--max-positions", "0",
        "--cut", str(params.cut),
        "--rollouts", str(params.rollouts),
        "--confirm-rollouts", str(params.confirm_rollouts),
        "--confirm-picks", str(params.confirm_picks),
        "--solve-max-unseen", str(params.solve_max_unseen),
        "--seed", str(survey_seed(work_dir)),
        "--threads", str(threads),
    ]
    # fmt: on
    if params.face_up_leaves:
        cmd.append("--open-leaves")
    return subprocess.run(cmd, capture_output=False).returncode


def deliver_surveyed(sink, work_dir: Path) -> tuple[int, int, float]:
    """Deliver every finished game in `work_dir` -- its slimmed survey file, then
    the .gcg of each position it kept -- and clear the game's files away.
    Returns (positions found, bytes, seconds)."""
    found, nbytes, t0 = 0, 0, time.monotonic()
    for survey in sorted(work_dir.glob(f"*{SURVEY_SUFFIX}")):
        stem = survey.name.removesuffix(SURVEY_SUFFIX)
        kept = slim_survey_file(survey)
        for game, turn in kept:
            name = gcg_name((stem, game, turn))
            nbytes += sink.deliver(work_dir / GCG_DIR / name, f"{GCG_DIR}/{name}")
        nbytes += sink.deliver(survey, f"{SURVEY_DIR}/{survey.name}")
        found += len(kept)
        (work_dir / f"{stem}.slog").unlink()
    shutil.rmtree(work_dir / GCG_DIR, ignore_errors=True)
    return found, nbytes, time.monotonic() - t0


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: finish and deliver whatever a previous run left in
    the work dir, then a game a cycle until max_cycles or SIGTERM."""
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = WorkerStats(ctx)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): surveying tag '{ctx.tag}' with {ctx.params}")

    cycle = 0
    try:
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            t0 = time.monotonic()
            if not any(work_dir.glob("*.slog")):  # else: a game a stopped run left unfinished
                play_game(work_dir, ctx.worker_id, ctx.params)
            t1 = time.monotonic()
            code = run_survey_tool(work_dir, ctx.params, ctx.threads)
            if code != 0:
                return code
            t2 = time.monotonic()
            found, nbytes, secs = deliver_surveyed(ctx.sink, work_dir)
            stats.cycle_done(
                {"gen_s": t1 - t0, "sim_s": t2 - t1, "upload_s": secs}, units=found, nbytes=nbytes
            )
            print(f"cycle {cycle}: {found} position(s) found")
    except WorkerStopped:
        print("SIGTERM: the game in progress resumes from its partial survey on restart")
    return 0


def positions_found(data_dir: Path) -> int:
    return sum(1 for _ in (data_dir / GCG_DIR).glob("*.gcg"))


def tick(spec: WorkloadSpec, task, hooks):
    """The scheduler entry: park the surveyors once the tag holds its target. Only
    the controller sees the whole store -- a rented worker delivers to a bucket
    and cannot count it -- so the stop is a gate from here rather than an exit
    from there. A parked worker is a paused container, which the dashboard's
    idle policy reads as nothing running: the rented machines stop themselves
    ten minutes later."""
    target = params_mod.validate(spec.params_cls, task.params).target_positions
    found = positions_found(spec.paths(task.tag).data_dir)
    reached = target > 0 and found >= target
    hooks.gate("generate", f"target reached: {found} of {target} positions" if reached else None)


def survey_dirs(tag: str) -> tuple[Path, Path]:
    """The tag's (survey files, .gcg exports) directories, for the viewer and the
    collection script."""
    data_dir = SPEC.paths(tag).data_dir
    return data_dir / SURVEY_DIR, data_dir / GCG_DIR


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    data_dir = spec.paths(tag).data_dir
    return [
        ("positions found", positions_found(data_dir)),
        ("games surveyed", sum(1 for _ in (data_dir / SURVEY_DIR).glob(f"*{SURVEY_SUFFIX}"))),
    ]


SPEC = WorkloadSpec(
    name="blind_spots",
    title="Collect HastyBot blind spots",
    params_cls=BlindSpotsParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Surveyor",
            runner="scribblez.workloads.blind_spots:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            stats=StatsSpec(
                unit="positions",
                phases={"gen_s": "self-play", "sim_s": "survey", "upload_s": "upload"},
            ),
        ),
    ),
    scheduler="scribblez.workloads.blind_spots:tick",
    progress="scribblez.workloads.blind_spots:progress",
    primary_params=("target_positions",),
    sync_data_dirs=(SURVEY_DIR, GCG_DIR),
)
