"""The match-arms workload: several named agent configurations ("arms"), each
played against one fixed opponent under the shared match discipline
(docs/evaluation_plan.md).

A tag is one experiment. Its frozen params name the arms (player-0 specs), the
opponent, and the pair budget. Every arm uses the same base seed, so the
engine's --paired mode gives common random numbers across arms: each faces
identical deals, and per-arm scores differ only by what the arms do with them.
A parameter sweep of one agent and a comparison of agents against baselines
are both just arm lists.

The singleton arms role plays one unmeasured arm per cycle and records its
result in the tag's dashboard.db. A restarted or respawned worker skips the
arms already measured, so once the batch is done a respawn exits at once.
"""

from dataclasses import dataclass

from scribblez.dashboard import db
from scribblez.params import ParamsError, param
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkloadSpec


@dataclass(frozen=True)
class Arm:
    """One named player-0 configuration of the experiment."""

    name: str
    player_spec: str


def parse_arms(text: str) -> list[Arm]:
    """Parse the `arms` param: semicolon-separated `name=<player spec>` entries,
    each split on its first '=' only, since player specs contain '=' themselves
    (e.g. "k5=--type=neural-sim --sim-top-k=5").

    Raises ParamsError on a malformed entry or a duplicate or empty name, so a
    bad string fails task creation instead of wedging the tag's runner. An
    empty string is a valid experiment with nothing to measure: the params
    dataclass must be constructible from its defaults.
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
        25,
        "pairs per play_game run: the granularity of progress lines, and the most a SIGTERM loses",
    )
    seed: int = param(
        1,
        "base game seed, shared across arms so every arm faces identical deals (must be nonzero)",
    )
    face_up_leaves: bool = param(True, "play the face-up-leaves variant (docs/roadmap.md)")

    # Runs wherever params are created (task creation, CLI, worker env), so a
    # bad experiment definition never reaches a runner.
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
    task_params = _task_params(spec, tag)
    if task_params is None:
        return []
    total = len(parse_arms(task_params.arms))
    db_path = spec.paths(tag).dashboard_db
    done = len(db.read_all_match_arms(db.connect(db_path))) if db_path.is_file() else 0
    return [("arms", f"{done}/{total} measured")]


def _task_params(spec: WorkloadSpec, tag: str):
    from scribblez.dashboard import tasks  # tasks imports the workload registry

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
