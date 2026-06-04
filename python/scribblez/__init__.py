"""scribblez: Python interface for the Scribblez Scrabble engine.

This package currently exposes a torch ``Dataset`` and ``DataLoader`` helpers
for consuming GCG game logs emitted by the C++ ``play_game`` binary. A
binary log format and FFI bindings will replace the GCG path later.
"""

from .dataset import GameLogDataset, TurnSample, build_dataloader

__all__ = ["GameLogDataset", "TurnSample", "build_dataloader"]
