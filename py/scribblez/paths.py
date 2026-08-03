"""Filesystem layout for a workload tag.

Tags are namespaced by workload, so each workload's runs are isolated (and the
dashboard's tag selector shows only its own workload's tags). Every artifact tied
to a tag lives under a single per-tag root, `<mount_root>/tags/<task>/<tag>/`:

    tags/<task>/<tag>/
      task.json                   frozen params + worker slots (master dashboard)
      logs/                       per-worker process logs
      stats/                      per-worker stats records (the Stats tab)
      params/                     per-worker provenance manifests
      train_state.json            the trainer's published cursor (tiny, atomic)
      data/
        staging/                  generator chunks awaiting generation assignment
        work/<worker_id>/         a generator's private in-progress cycle output
        test/                     *.slog  -- frozen held-out games (+ manifest.json)
        generations/gen_NNNNNN/   *.slog  -- one generation (+ manifest.json)
      checkpoints/                model.pt (rolling resume checkpoint)
      models/                     model_epoch_XXXX.onnx
      dashboard.db                metrics + eval data (rendered by the dashboard)

Workloads use the subset of this tree they need (kill_test keeps its pairs under
data/slogs/; only the training workloads have checkpoints or a dashboard DB).
This module is the single source of truth for the layout: scripts derive every
path from a `TagPaths` rather than reassembling subdirectories.

All training/eval results are written to dashboard.db (a SQLite store); the
dashboard renders every plot on the fly from it -- no PNG artifacts.
"""

from pathlib import Path

DEFAULT_MOUNT_ROOT = Path("/workspace/mount")

# Workload identifiers: the `<task>` level of the tags/ tree, the workload
# registry keys, and the dashboard's task slugs. The single source of truth.
POSITION_EVAL = "position_eval"
MAX_MOVE_PER_LANE = "max_move_per_lane"
KILL_TEST = "kill_test"
MOVE_SET_EVAL = "move_set_eval"


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
        """Generator chunks land here (directly, or via cloud sync) until the
        generation scheduler assigns them to a generation directory."""
        return self.data_dir / "staging"

    def work_dir(self, worker_id: str) -> Path:
        """A generator's private scratch dir for the in-progress cycle. Wiped on
        worker start: a crash mid-cycle may leave a truncated .slog here, so
        leftovers are never delivered."""
        return self.data_dir / "work" / worker_id

    @property
    def train_dir(self) -> Path:
        """One-shot training data (the scripts/generate_data.py flow); the
        generational pipeline uses generations_dir instead."""
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def generations_dir(self) -> Path:
        """Parent of the per-generation game directories. Each child
        `gen_<NNNNNN>/` holds one generation's .slog files plus a manifest; the
        trainer trains over a sliding window of the most recent complete
        generations."""
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
    def train_state_path(self) -> Path:
        """The trainer's published cursor: a tiny JSON the generation scheduler
        and the dashboard read instead of parsing the torch checkpoint."""
        return self.root / "train_state.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def onnx_dir(self) -> Path:
        return self.root / "models"

    @property
    def dashboard_db(self) -> Path:
        """SQLite store of all metrics + eval data, read by the dashboard."""
        return self.root / "dashboard.db"

    def checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoints_dir / f"model_epoch_{epoch:04d}.pt"

    @property
    def rolling_checkpoint(self) -> Path:
        """Single .pt reused across the streaming run (full resume state)."""
        return self.checkpoints_dir / "model.pt"

    def onnx_path(self, epoch: int) -> Path:
        return self.onnx_dir / f"model_epoch_{epoch:04d}.onnx"
