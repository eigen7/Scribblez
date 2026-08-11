"""The match-arms workload: a batch of named agent configurations, each played
against one fixed opponent under the shared match discipline (roadmap A4/E2).

A tag is one experiment: its frozen params name the arms (player-0 specs), the
opponent, and the pair budget, and every arm shares the same base seed, so the
engine's --paired mode gives cross-arm common random numbers -- every arm faces
identical deals, and per-arm scores differ only by what the arms do with them.
The A4 sensitivity sweep (neural-sim --sim-top-k / --drop-best-prob arms) and
the move-set-evaluation agent's baseline comparisons are both just arm lists.

The singleton arms role plays one unmeasured arm per cycle and writes its
match_arm row; a restart (or the reconciler's respawn of an exited worker)
skips the arms already measured, so finishing the batch makes further respawns
cheap no-ops (the pair_store idempotency shape).
"""

from dataclasses import dataclass

from scribblez.params import ParamsError, param
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkloadSpec


@dataclass(frozen=True)
class Arm:
    """One named player-0 configuration of the experiment."""

    name: str
    player_spec: str


def parse_arms(text: str) -> list[Arm]:
    """Parse the `arms` param: semicolon-separated `name=<player spec>` entries,
    each split on its FIRST '=' only, since player specs contain '=' themselves
    (e.g. "k5=--type=neural-sim --sim-top-k=5"). Raises ParamsError on a
    malformed entry or a duplicate or empty name -- surfaced at task creation,
    where params freeze; a bad string must not become a permanently wedged tag
    the runner rediscovers every cycle. An empty string is a valid empty
    experiment (params must stay default-constructible for the registry
    contract); the runner then simply has nothing to measure.
    """
    arms: list[Arm] = []
    errors: list[str] = []
    entries = [e.strip() for e in text.split(";") if e.strip()]
    for entry in entries:
        name, eq, spec = entry.partition("=")
        name, spec = name.strip(), spec.strip()
        if not eq or not name or not spec:
            errors.append(f"arms: expected 'name=<player spec>', got '{entry}'")
        elif name.startswith("-"):
            # "--type=greedy" would parse as name "--type": almost certainly a
            # forgotten name, so refuse rather than record a nonsense arm.
            errors.append(f"arms: name '{name}' looks like a spec token -- missing 'name='?")
        elif any(a.name == name for a in arms):
            errors.append(f"arms: duplicate arm name '{name}'")
        else:
            arms.append(Arm(name, spec))
    if errors:
        raise ParamsError(*errors)
    return arms


@dataclass(frozen=True)
class MatchArmsParams:
    arms: str = param(
        "",
        "semicolon-separated 'name=<player spec>' arms (player 0), each spec split from its "
        "name at the first '='; e.g. 'k5=--type=neural-sim --model=m.onnx --sim-top-k=5; "
        "k10=--type=neural-sim --model=m.onnx'",
    )
    opponent: str = param("--type=sim", "the fixed opponent's --player spec, shared by every arm")
    pairs_per_arm: int = param(200, "mirrored game pairs played per arm")
    round_pairs: int = param(
        25, "pairs per play_game invocation (the granularity of progress lines and SIGTERM loss)"
    )
    seed: int = param(
        1,
        "base game seed, shared across arms so every arm faces identical deals (must be nonzero)",
    )
    face_up_leaves: bool = param(False, "play the face-up-leaves variant (docs/roadmap.md)")

    # Frozen params are validated where they are created (task creation, CLI,
    # worker env), so a bad experiment definition can never reach a runner.
    def __post_init__(self):
        parse_arms(self.arms)
        if self.seed == 0:
            raise ParamsError("seed must be nonzero (0 asks play_game for entropy, unfixing deals)")
        if self.pairs_per_arm < 1:
            raise ParamsError("pairs_per_arm must be >= 1")
        if self.round_pairs < 1:
            raise ParamsError("round_pairs must be >= 1")


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    """Arms measured / total, read the same way the runner decides what is left."""
    from scribblez.dashboard import db  # heavy-ish import kept out of module load

    task_params = _task_params(spec, tag)
    if task_params is None:
        return []
    total = len(parse_arms(task_params.arms))
    db_path = spec.paths(tag).dashboard_db
    done = len(db.read_all_match_arms(db.connect(db_path))) if db_path.is_file() else 0
    return [("arms", f"{done}/{total} measured")]


def _task_params(spec: WorkloadSpec, tag: str):
    from scribblez.dashboard import tasks  # local import: avoid a module cycle

    task = tasks.load_task(spec, tag)
    return None if task is None else MatchArmsParams(**task.params)


SPEC = WorkloadSpec(
    name="match_arms",
    title="Match arms (agent comparison)",
    params_cls=MatchArmsParams,
    roles=(
        RoleSpec(
            name="arms",
            title="Arms runner (GPU)",
            runner="scribblez.match_eval.arms:run",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(unit="games", phases={"match_s": "match play"}),
        ),
    ),
    progress="scribblez.workloads.match_arms:progress",
)
