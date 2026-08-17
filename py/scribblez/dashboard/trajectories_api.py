"""The evidence_trajectories Trajectories tab's data plane: a hand-maintained
.gcg position set (positions/<lexicon>/<set>/) simmed under THIS tag's
proposer and recipe, and the tag's evidence checkpoints re-scoring each
position at every evidence prefix (scribblez.evidence.trajectory_view).
Registered alongside the rest of the data plane by api.make_app().

Routes (all take task + tag):
  /api/evidence_trajectories/sets         the position sets on disk
  /api/evidence_trajectories/positions    ?set= -> the set's positions
  /api/evidence_trajectories/generations  the tag's checkpoints: 0 is the frozen
                                          student itself, N its pass-N checkpoint
  /api/evidence_trajectories/position     ?set=&position=&generation=&prefix=&slot=
                                          -> the board bundle + the model view

The first request for a (set, tag) generates the set's trajectory sidecars
(position_sets.ensure_sobs: cached under <mount>/cache/trajectory_sets/, so
only a new position or a changed .gcg sims again -- a few positions at 200
rollouts take well under a minute, during which the request blocks). Model
work runs on CPU in torch: a checkpoint's plain pass over a position's few
hundred legal moves takes ~0.1s, a prefix's conditioned pass ~10ms, and
both are memoized per (checkpoint file, position), keyed by mtime so a
rewritten file (the student's rolling checkpoint under a live mset trainer, a
regenerated sidecar) is re-analyzed.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tornado.ioloop
import tornado.web

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.evidence.checkpoints import EvidenceCheckpoint, load_evidence_checkpoint
from scribblez.evidence.trajectory_view import DecisionAnalysis, payload
from scribblez.ffi import gcg_position_board_json
from scribblez.paths import EVIDENCE_TRAJECTORIES, REPO_ROOT, TagPaths
from scribblez.sim_evidence.position_sets import TrajectoryRecipe, ensure_sobs, set_gcgs
from scribblez.sim_evidence.sobs import SobsPosition, read_sobs
from scribblez.workloads.evidence_trajectories import EvidenceTrajectoriesParams, max_evidence

POSITIONS_ROOT = REPO_ROOT / "positions" / "NWL23"
DEFAULT_SET = "face-up-trajectory-set"
_CHECKPOINT_RE = re.compile(r"model_epoch_(\d{4})\.pt$")
# Threads for on-demand sidecar generation (the sims are the long pole).
_SIM_THREADS = max(4, (os.cpu_count() or 8) - 4)
# The position handler's work -- sidecar generation, checkpoint loads, model
# passes -- runs off the IOLoop so a first request does not stall the rest of
# the dashboard; one worker serializes the model and cache access.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def position_sets() -> list[str]:
    """Directories under positions/NWL23/ holding .gcg files, the default set first."""
    if not POSITIONS_ROOT.is_dir():
        return []
    names = sorted(d.name for d in POSITIONS_ROOT.iterdir() if d.is_dir() and set_gcgs(d))
    if DEFAULT_SET in names:
        names.remove(DEFAULT_SET)
        names.insert(0, DEFAULT_SET)
    return names


def set_dir(name: str) -> Path:
    """The set's directory; a name that is not a plain directory name (path
    separators, dots) or does not exist is a KeyError."""
    if not name or "/" in name or name.startswith("."):
        raise KeyError(f"bad position set {name!r}")
    d = POSITIONS_ROOT / name
    if not d.is_dir():
        raise KeyError(f"no position set {name!r}")
    return d


def tag_params(tag: str) -> EvidenceTrajectoriesParams:
    """The tag's frozen params (its task.json)."""
    task = tasks.load_task(workloads.get(EVIDENCE_TRAJECTORIES), tag)
    if task is None:
        raise KeyError(f"tag {tag!r} has no task.json")
    return params_mod.validate(EvidenceTrajectoriesParams, task.params)


def recipe_of(params: EvidenceTrajectoriesParams) -> TrajectoryRecipe:
    return TrajectoryRecipe(
        rollouts=params.rollouts,
        proposals_min=params.proposals_min,
        proposals_max=params.proposals_max,
        temperature=params.temperature,
        proposal_pool=params.proposal_pool,
        open_leaves=params.face_up_leaves,
    )


def generations(paths: TagPaths, params: EvidenceTrajectoriesParams) -> list[dict]:
    """The generation slider's stops: 0 for the frozen student (when its
    checkpoint exists), then every per-pass checkpoint on disk as generation
    epoch + 1."""
    gens = []
    if params.student_checkpoint and os.path.isfile(params.student_checkpoint):
        gens.append({"generation": 0, "epoch": None, "path": params.student_checkpoint})
    if paths.checkpoints_dir.is_dir():
        for f in sorted(paths.checkpoints_dir.iterdir()):
            m = _CHECKPOINT_RE.search(f.name)
            if m:
                epoch = int(m.group(1))
                gens.append({"generation": epoch + 1, "epoch": epoch, "path": str(f)})
    return gens


def sidecars(set_name: str, params: EvidenceTrajectoriesParams, mount_root) -> dict[str, Path]:
    """{stem: .sobs} for the set under the tag's proposer + recipe, generating
    what is missing (blocking)."""
    return ensure_sobs(
        set_dir(set_name), Path(params.proposer_model), recipe_of(params), _SIM_THREADS, mount_root
    )


@functools.lru_cache(maxsize=4)
def _checkpoint(path: str, mtime_ns: int) -> EvidenceCheckpoint:
    """Keyed by mtime: a per-pass checkpoint is written once, but generation
    0's student rolling checkpoint is rewritten while its trainer runs."""
    return load_evidence_checkpoint(path, "cpu")


@functools.lru_cache(maxsize=64)
def _board_bundle(gcg_path: str, open_leaves: bool) -> dict:
    return gcg_position_board_json(Path(gcg_path).read_text(), open_leaves)


@functools.lru_cache(maxsize=64)
def _sobs_position(sobs_path: str, mtime_ns: int) -> SobsPosition:
    """The set position's single trajectory (keyed by mtime: a sidecar is
    regenerated in place when its .gcg changes)."""
    return read_sobs(sobs_path)[0]


@functools.lru_cache(maxsize=64)
def _analysis(
    ckpt_path: str,
    ckpt_mtime_ns: int,
    gcg_path: str,
    sobs_path: str,
    sobs_mtime_ns: int,
    max_e: int,
) -> DecisionAnalysis:
    """The prefix-independent passes of (checkpoint, position), memoized; the
    conditioned passes memoize inside it."""
    ckpt = _checkpoint(ckpt_path, ckpt_mtime_ns)
    sobs = _sobs_position(sobs_path, sobs_mtime_ns)
    return DecisionAnalysis(ckpt, Path(gcg_path).read_text(), sobs, max_e, "cpu")


def _mtime(path) -> int:
    return os.stat(path).st_mtime_ns


def position_payload(
    tag: str, task: str, mount_root, set_name: str, position: int, generation: int, prefix, slot
) -> dict:
    """The tab's per-position view. `prefix` None (or "last") means the
    largest evidence prefix; `slot` None means the last candidate in the prefix
    (or the anchor at prefix 0)."""
    params = tag_params(tag)
    paths = TagPaths(tag, task, mount_root)
    gcgs = set_gcgs(set_dir(set_name))
    if not 0 <= position < len(gcgs):
        raise KeyError("position out of range")
    gens = {g["generation"]: g["path"] for g in generations(paths, params)}
    if generation not in gens:
        raise KeyError(f"no checkpoint for generation {generation}")
    gcg = gcgs[position]
    sobs_path = sidecars(set_name, params, mount_root)[gcg.stem]
    ckpt_path = gens[generation]
    analysis = _analysis(
        ckpt_path,
        _mtime(ckpt_path),
        str(gcg),
        str(sobs_path),
        _mtime(sobs_path),
        max_evidence(params),
    )
    sobs = analysis.sobs
    max_prefix = max(sobs.evidence_prefix_sizes())
    p = max_prefix if prefix is None else min(max(int(prefix), 0), max_prefix)
    s = (max(p - 1, 0) if slot is None else int(slot)) if len(sobs.moves) else None
    bundle = _board_bundle(str(gcg), params.face_up_leaves)
    view = payload(analysis, bundle["moves"], p, s)
    board = {k: v for k, v in bundle.items() if k != "moves"}
    return {
        "name": gcg.stem,
        "set": set_name,
        "generation": generation,
        "board": board,
        **view,
    }


class _Base(tornado.web.RequestHandler):
    @property
    def mount_root(self) -> str:
        return self.settings["mount_root"]

    def fail(self, status: int, message: str):
        self.set_status(status)
        self.write({"error": message})


class SetsHandler(_Base):
    def get(self):
        self.write({"sets": position_sets(), "default": DEFAULT_SET})


class PositionsHandler(_Base):
    def get(self):
        try:
            gcgs = set_gcgs(set_dir(self.get_query_argument("set", DEFAULT_SET)))
        except KeyError as e:
            self.fail(404, str(e))
            return
        self.write({"positions": [g.stem for g in gcgs]})


class GenerationsHandler(_Base):
    """The generation slider's stops for a tag: [{generation}], 0 = the frozen
    student, N = the trainer's pass N-1 checkpoint."""

    def get(self):
        tag, task = self.get_query_argument("tag"), self.get_query_argument("task")
        try:
            params = tag_params(tag)
        except KeyError:
            self.write({"generations": []})
            return
        gens = generations(TagPaths(tag, task, self.mount_root), params)
        self.write(
            {"generations": [{"generation": g["generation"], "epoch": g["epoch"]} for g in gens]}
        )


class PositionHandler(_Base):
    """Board + trajectory + model view; see position_payload. `generation` may
    be 'latest'; `prefix` defaults to the largest; `slot` to the prefix's last
    candidate."""

    async def get(self):
        tag, task = self.get_query_argument("tag"), self.get_query_argument("task")
        set_name = self.get_query_argument("set", DEFAULT_SET)
        try:
            params = tag_params(tag)
            gens = generations(TagPaths(tag, task, self.mount_root), params)
            gen_arg = self.get_query_argument("generation", "latest")
            if not gens:
                self.fail(404, "no checkpoints yet")
                return
            generation = gens[-1]["generation"] if gen_arg in ("", "latest") else int(gen_arg)
            prefix = self.get_query_argument("prefix", None)
            slot = self.get_query_argument("slot", None)
            build = functools.partial(
                position_payload,
                tag,
                task,
                self.mount_root,
                set_name,
                int(self.get_query_argument("position", "0")),
                generation,
                None if prefix in (None, "", "last") else prefix,
                None if slot in (None, "") else slot,
            )
            self.write(await tornado.ioloop.IOLoop.current().run_in_executor(_EXECUTOR, build))
        except KeyError as e:
            self.fail(404, str(e))
        except ValueError as e:  # bad arg / mismatched sidecar / gcg parse
            self.fail(400, str(e))
        except (OSError, subprocess.CalledProcessError) as e:  # engine / generator unavailable
            self.fail(503, str(e))


ROUTES = [
    (r"/api/evidence_trajectories/sets", SetsHandler),
    (r"/api/evidence_trajectories/positions", PositionsHandler),
    (r"/api/evidence_trajectories/generations", GenerationsHandler),
    (r"/api/evidence_trajectories/position", PositionHandler),
]
