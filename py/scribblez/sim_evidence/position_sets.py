"""Trajectory sidecars for .gcg position sets, generated on demand and cached.

A position set is a directory of hand-maintained .gcg files (positions/<lexicon>/
<set>/), each read at its final recorded state with the side to move's rack
from its #RackN pragma (engine: read_gcg_position, data/gcg_reader.h). Its trajectory .sobs -- what the dashboard's
trajectory pane shows and what the evidence trainer's position-set metric
reads -- depend on the proposer and the recipe, so they are not committed:
they live under <mount>/cache/trajectory_sets/<set>/<key>/, keyed by the
proposer's bytes and the recipe, and each is regenerated when its .gcg
changes. `ensure_sobs` is the one entry point: it runs
evidence_trajectory_generator --gcg for whatever is missing or stale and
returns the sidecar per position.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from scribblez.paths import DEFAULT_MOUNT_ROOT, ENGINE_DIR

TRAJECTORY_GENERATOR = ENGINE_DIR / "evidence_trajectory_generator"
CACHE_DIR = "cache/trajectory_sets"
MANIFEST = "manifest.json"  # stem -> sha256 of the .gcg the sidecar was generated from


@dataclass(frozen=True)
class TrajectoryRecipe:
    """The generator's per-position recipe plus the information condition --
    everything besides the proposer that determines a trajectory."""

    rollouts: int = 200
    proposals_min: int = 2
    proposals_max: int = 8
    temperature: float = 0.05
    proposal_pool: int = 64
    open_leaves: bool = False
    seed: int = 0

    def args(self) -> list[str]:
        return [
            f"--rollouts={self.rollouts}",
            f"--proposals-min={self.proposals_min}",
            f"--proposals-max={self.proposals_max}",
            f"--temperature={self.temperature}",
            f"--proposal-pool={self.proposal_pool}",
            f"--seed={self.seed}",
            *(["--open-leaves"] if self.open_leaves else []),
        ]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_key(proposer_model: Path, recipe: TrajectoryRecipe) -> str:
    """The cache subdirectory for (proposer, recipe): a hash of the model's
    bytes and the recipe's fields, so a re-exported model or a changed knob
    never reads another configuration's sidecars."""
    h = hashlib.sha256(Path(proposer_model).read_bytes())
    h.update(json.dumps(asdict(recipe), sort_keys=True).encode())
    return h.hexdigest()[:16]


def set_gcgs(set_dir: Path) -> list[Path]:
    return sorted(Path(set_dir).glob("*.gcg"))


def cache_dir(
    set_dir: Path, proposer_model: Path, recipe: TrajectoryRecipe, mount_root=None
) -> Path:
    root = Path(mount_root or DEFAULT_MOUNT_ROOT)
    return root / CACHE_DIR / Path(set_dir).name / cache_key(proposer_model, recipe)


def _load_manifest(d: Path) -> dict:
    f = d / MANIFEST
    return json.loads(f.read_text()) if f.is_file() else {}


def _stale_or_missing(gcgs: list[Path], d: Path) -> list[Path]:
    """The .gcg files whose sidecar is absent or was generated from other bytes.
    Stale sidecars are removed so the generator (which skips existing outputs)
    regenerates them."""
    manifest = _load_manifest(d)
    pending = []
    for gcg in gcgs:
        sobs = d / f"{gcg.stem}.sobs"
        if sobs.exists() and manifest.get(gcg.stem) == _sha256(gcg.read_bytes()):
            continue
        sobs.unlink(missing_ok=True)
        pending.append(gcg)
    return pending


def ensure_sobs(
    set_dir: Path,
    proposer_model: Path,
    recipe: TrajectoryRecipe,
    threads: int,
    mount_root=None,
) -> dict[str, Path]:
    """Every position's trajectory sidecar for (set, proposer, recipe),
    generating what is missing or stale. Returns {gcg stem: .sobs path}."""
    gcgs = set_gcgs(set_dir)
    d = cache_dir(set_dir, proposer_model, recipe, mount_root)
    d.mkdir(parents=True, exist_ok=True)
    pending = _stale_or_missing(gcgs, d)
    if pending:
        cmd = [
            str(TRAJECTORY_GENERATOR),
            *[f"--gcg={g}" for g in pending],
            f"--out-dir={d}",
            f"--model={proposer_model}",
            f"--threads={threads}",
            *recipe.args(),
        ]
        subprocess.run(cmd, check=True)
        manifest = _load_manifest(d)
        manifest.update({g.stem: _sha256(g.read_bytes()) for g in pending})
        (d / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return {g.stem: d / f"{g.stem}.sobs" for g in gcgs}
