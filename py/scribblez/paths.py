"""Filesystem layout for a training tag.

Every artifact tied to a tag lives under a single per-tag root,
`<mount_root>/tags/<tag>/`, organized as:

    tags/<tag>/
      data/train/                 *.slog  -- training games
      data/test/                  *.slog  -- frozen held-out games
      checkpoints/                model_epoch_XXXX.pt
      models/                     model_epoch_XXXX.onnx
      monotonicity-probe-analysis/  gen-XXXX.png
      score-belief-probe-analysis/  gen-XXXX.png
      test-subset/                positions.slog + pos-XX.txt  (frozen eval set)
      metrics.jsonl               per-epoch training + eval metrics

This module is the single source of truth for that layout: scripts derive
every path from a `TagPaths` rather than reassembling subdirectories.
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
    def probe_analysis_dir(self) -> Path:
        return self.root / "monotonicity-probe-analysis"

    @property
    def score_belief_dir(self) -> Path:
        return self.root / "score-belief-probe-analysis"

    @property
    def test_subset_dir(self) -> Path:
        """Frozen evaluation positions sampled from the test split (standard .slog)."""
        return self.root / "test-subset"

    @property
    def test_subset_slog(self) -> Path:
        return self.test_subset_dir / "positions.slog"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    def checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoints_dir / f"model_epoch_{epoch:04d}.pt"

    def onnx_path(self, epoch: int) -> Path:
        return self.onnx_dir / f"model_epoch_{epoch:04d}.onnx"

    def probe_image_path(self, generation: int) -> Path:
        return self.probe_analysis_dir / f"gen-{generation:04d}.png"

    def score_belief_image_path(self, generation: int) -> Path:
        return self.score_belief_dir / f"gen-{generation:04d}.png"

    def position_dump_path(self, index: int) -> Path:
        return self.test_subset_dir / f"pos-{index:02d}.txt"
