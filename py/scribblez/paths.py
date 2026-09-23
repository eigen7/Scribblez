"""Filesystem layout for a workload tag.

A tag is one run of a workload. Tags are namespaced by workload, and every
artifact tied to a tag lives under one root, `<mount_root>/tags/<task>/<tag>/`:

    tags/<task>/<tag>/
      task.json                   frozen params + worker slots (written by the dashboard)
      logs/                       per-worker process logs
      stats/                      per-worker stats records (the Stats tab)
      params/                     per-worker provenance manifests
      records/                    the trainer's run + per-generation records,
                                  ingested into dashboard.db by the controller
      controls.json               the operator's live controls, read by the trainer
      train_state.json            the trainer's published cursor (tiny, atomic)
      data/
        staging/                  generator chunks awaiting generation assignment
        work/<worker_id>/         a generator's private in-progress cycle output
        match_inbox/<worker_id>/  the model a match-eval slot was assigned
        match_results/            finished matches awaiting ingest into dashboard.db
        test/                     *.slog  -- frozen held-out games (+ manifest.json)
        generations/gen_NNNNNN/   *.slog  -- one generation (+ manifest.json)
      checkpoints/                model.pt (rolling resume checkpoint)
      models/                     model_epoch_XXXX.onnx
      dashboard.db                metrics + eval data (rendered by the dashboard)

Workloads use the subset of this tree they need (kill_test keeps its pairs under
data/slogs/; only the training workloads have checkpoints or a dashboard DB).
Code derives every path from a `TagPaths` rather than reassembling
subdirectories.

dashboard.db (SQLite) holds every training and eval result, and the dashboard
renders its plots from it. Only the dashboard writes it: the trainer delivers
records under records/ and the dashboard ingests them (the protocol is
documented in generational/records.py).
"""

from pathlib import Path

DEFAULT_MOUNT_ROOT = Path("/workspace/mount")

# The checkout this package was imported from, so code running in a git
# worktree uses the worktree's own binaries and data.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "target" / "engine"

# The position-evaluation eval sets (position_eval/analysis.py): the small
# hand-built set the dashboard's Positions tab browses, and the larger
# machine-harvested set behind the Loss tab's aggregate quality curves. A
# worker running from a deployed bundle rather than a checkout fetches them
# separately (py/cloud/worker_deps.py).
EVAL_POSITIONS_DIRS = (
    REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset",
    REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset-large",
)

# Workload identifiers: the `<task>` level of the tags/ tree, the workload
# registry keys, and the dashboard's task slugs.
POSITION_EVAL = "position_eval"
MAX_MOVE_PER_LANE = "max_move_per_lane"
KILL_TEST = "kill_test"
MOVE_SET_EVAL = "move_set_eval"
EVIDENCE_TRAJECTORIES = "evidence_trajectories"
MATCH_ARMS = "match_arms"

# The data/ subdirectories of the match-eval exchange. They are defined here
# because both ends use them: the controller writes an inbox the worker polls,
# and the worker delivers results the controller ingests.
MATCH_INBOX_DIR = "match_inbox"
MATCH_RESULTS_DIR = "match_results"

# Filename prefix of a per-generation ONNX export; it distinguishes exports
# from the shared blobs beside them in models/ (see onnx_sidecars).
ONNX_PREFIX = "model_epoch_"

# What a trainer on a rented machine delivers through the results bucket,
# relative to the tag root, for scripts/cloud_sync.py to pull back. Records and
# exports never change once written, so they are synced by size alone; the
# rolling checkpoint and train_state.json are rewritten in place.
TRAINER_OUTPUT_DIRS = ("records", "models", "checkpoints")
TRAINER_OUTPUT_IMMUTABLE = ("records", "models")
TRAINER_OUTPUT_FILES = ("train_state.json",)

# The trainer's record stream (generational/records.py), relative to the tag
# root because the trainer writes through a results sink that maps them either
# under the local tag root or under the tag's prefix in the bucket.
RECORDS_DIR = "records"
RUN_RECORD_REL = f"{RECORDS_DIR}/run.json"
CONTROLS_REL = "controls.json"


def generation_record_rel(generation: int) -> str:
    return f"{RECORDS_DIR}/gen_{generation:06d}.json"


def generation_preds_rel(generation: int) -> str:
    return f"{RECORDS_DIR}/gen_{generation:06d}.npz"


# Appended to an inbox model's filename by the worker once it has played it:
# the worker stops offering it, while the controller still counts that
# generation as assigned (match_eval/dispatch.py).
DONE_SUFFIX = ".done"


class TagPaths:
    """Resolves every per-tag artifact path under `<mount_root>/tags/<task>/<tag>/`."""

    def __init__(self, tag: str, task: str, mount_root: str | Path = DEFAULT_MOUNT_ROOT):
        self.tag = tag
        self.task = task
        self.mount_root = Path(mount_root)

    @property
    def root(self) -> Path:
        return self.mount_root / "tags" / self.task / self.tag

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def staging_dir(self) -> Path:
        """Generator output waits here until the generation scheduler assigns
        it to a generation directory."""
        return self.data_dir / "staging"

    def work_dir(self, worker_id: str) -> Path:
        """A generator's scratch dir for its in-progress cycle. Wiped on worker
        start, because a crash mid-cycle can leave a truncated .slog here."""
        return self.data_dir / "work" / worker_id

    @property
    def train_dir(self) -> Path:
        """Training data from scripts/generate_data.py; the generational
        pipeline uses generations_dir instead."""
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def generations_dir(self) -> Path:
        """Parent of the per-generation `gen_NNNNNN/` directories, each holding
        one generation's .slog files plus a manifest. The trainer trains over a
        sliding window of the most recent complete generations."""
        return self.data_dir / "generations"

    def generation_dir(self, index: int) -> Path:
        return self.generations_dir / f"gen_{index:06d}"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def stats_dir(self) -> Path:
        return self.root / "stats"

    @property
    def records_dir(self) -> Path:
        """The trainer's delivered records, which the controller ingests into
        dashboard.db (generational/train_ingest.py)."""
        return self.root / RECORDS_DIR

    @property
    def run_record_path(self) -> Path:
        return self.root / RUN_RECORD_REL

    def generation_record_path(self, generation: int) -> Path:
        return self.root / generation_record_rel(generation)

    @property
    def controls_path(self) -> Path:
        """The operator's live controls: written by the dashboard, read by the
        trainer (generational/records.py)."""
        return self.root / CONTROLS_REL

    @property
    def train_state_path(self) -> Path:
        """The trainer's progress cursor, a small JSON that the generation
        scheduler and the dashboard read instead of loading the torch
        checkpoint."""
        return self.root / "train_state.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def onnx_dir(self) -> Path:
        return self.root / "models"

    @property
    def dashboard_db(self) -> Path:
        """SQLite store of all metrics and eval data, read by the dashboard."""
        return self.root / "dashboard.db"

    def checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoints_dir / f"model_epoch_{epoch:04d}.pt"

    @property
    def rolling_checkpoint(self) -> Path:
        """The trainer's full resume state, overwritten as training progresses."""
        return self.checkpoints_dir / "model.pt"

    def onnx_path(self, epoch: int) -> Path:
        return self.onnx_dir / f"{ONNX_PREFIX}{epoch:04d}.onnx"

    # The evidence trainer exports two graphs per generation: move_proposal_cache
    # at onnx_path (the generation's model, as far as exported_generations and
    # the match inbox are concerned) and its move_proposal_step companion under
    # step/. In unfrozen mode it also exports the plain student under plain/.
    # Subdirectories keep the companions out of the model_epoch_* glob and out
    # of onnx_sidecars.
    def proposal_step_path(self, epoch: int) -> Path:
        return self.onnx_dir / "step" / f"{ONNX_PREFIX}{epoch:04d}.onnx"

    def plain_onnx_path(self, epoch: int) -> Path:
        return self.onnx_dir / "plain" / f"{ONNX_PREFIX}{epoch:04d}.onnx"

    @staticmethod
    def onnx_epoch(path: Path) -> int:
        """The epoch an export's filename encodes -- the inverse of onnx_path."""
        return int(path.stem.rsplit("_", 1)[1])

    def exported_generations(self) -> list[int]:
        """The generations with an ONNX export in models/, ascending. The
        filesystem rather than the dashboard is the authority on what is
        deployable, because the export is written before any dashboard row
        records it."""
        return sorted(self.onnx_epoch(p) for p in self.onnx_dir.glob(f"{ONNX_PREFIX}*.onnx"))

    @property
    def onnx_sidecars(self) -> list[Path]:
        """Non-export files in models/ that exports reference: the shared
        external-data blob holding the frozen lexicon buffers
        (position_eval/onnx_export.py), written once for all generations. A
        model loads only beside its sidecars, so whatever ships a model must
        ship these too."""
        return sorted(
            p for p in self.onnx_dir.glob("*") if p.is_file() and not p.name.startswith(ONNX_PREFIX)
        )

    @property
    def match_results_dir(self) -> Path:
        """Finished matches delivered by match-eval workers, awaiting ingest
        into dashboard.db (match_eval/dispatch.py)."""
        return self.data_dir / MATCH_RESULTS_DIR

    def match_inbox_dir(self, worker_id: str) -> Path:
        """Where a match-eval slot receives the model it is to play
        (match_eval/dispatch.py)."""
        return self.data_dir / MATCH_INBOX_DIR / worker_id
