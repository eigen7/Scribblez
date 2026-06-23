"""Filesystem layout for a training tag.

Every artifact tied to a tag lives under a single per-tag root,
`<mount_root>/tags/<tag>/`, organized as:

    tags/<tag>/
      data/train/                 *.slog  -- training games
      data/test/                  *.slog  -- frozen held-out games
      checkpoints/                model_epoch_XXXX.pt
      models/                     model_epoch_XXXX.onnx
      test-subset/                positions.slog + pos-XX.txt  (frozen eval set)
      dashboard.db                metrics + eval data (rendered by the dashboard)

This module is the single source of truth for that layout: scripts derive
every path from a `TagPaths` rather than reassembling subdirectories.

All training/eval results are written to dashboard.db (a SQLite store); the
Bokeh dashboard renders every plot on the fly from it -- no PNG artifacts.
"""

from pathlib import Path

DEFAULT_MOUNT_ROOT = Path("/workspace/mount")


class TagPaths:
    """Resolves every per-tag artifact path under `<mount_root>/tags/<tag>/`."""

    def __init__(self, tag: str, mount_root: str | Path = DEFAULT_MOUNT_ROOT):
        self.tag = tag
        self.mount_root = Path(mount_root)

    @property
    def root(self) -> Path:
        return self.mount_root / "tags" / self.tag

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def onnx_dir(self) -> Path:
        return self.root / "models"

    @property
    def test_subset_dir(self) -> Path:
        """Frozen evaluation positions sampled from the test split (standard .slog)."""
        return self.root / "test-subset"

    @property
    def test_subset_slog(self) -> Path:
        return self.test_subset_dir / "positions.slog"

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

    def position_dump_path(self, index: int) -> Path:
        return self.test_subset_dir / f"pos-{index:02d}.txt"
