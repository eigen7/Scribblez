"""Quick smoke test for the GameLogDataset.

Usage:
    python -m scripts.inspect_log path/to/game.json [more.json ...]
"""

from __future__ import annotations

import sys

import torch

from scribblez import GameLogDataset, build_dataloader


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    ds = GameLogDataset(argv[1:])
    print(f"Loaded {len(ds.game_paths)} game log(s) -> {len(ds)} turn samples")

    loader = build_dataloader(argv[1:], batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print("Batch tensor shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: list[{len(v)}] (e.g. {v[0]!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
