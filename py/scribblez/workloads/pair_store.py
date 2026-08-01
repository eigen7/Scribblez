"""Sidecar-pair delivery to a tag's data store, shared by the workloads whose
product is a .slog plus a same-stem sidecar (.sobs for kill_test, .mset for
move_set_eval).

Both members of a pair get the same -<worker_id> stem suffix, so names stay
globally unique across workers while preserving the stem-based pair matching
downstream readers rely on. The sidecar is delivered before its .slog: a .slog
missing its sidecar reads as pending work downstream, while an orphaned
sidecar is inert -- so the store only ever presents complete pairs plus inert
leftovers.
"""

import time
from pathlib import Path


def deliver_pairs(
    sink, out_dir: Path, worker_id: str, sidecar_ext: str, dest_dir: str
) -> tuple[int, int, float]:
    """Deliver every complete .slog/sidecar pair in `out_dir` (not just the
    current cycle's -- a restarted worker flushes leftovers too) to the tag's
    `dest_dir` store. Returns (pairs, bytes, seconds)."""
    moved, nbytes, t0 = 0, 0, time.monotonic()
    for sidecar in sorted(out_dir.glob(f"*{sidecar_ext}")):
        slog = sidecar.with_suffix(".slog")
        for f in (sidecar, slog):
            nbytes += sink.deliver(f, f"{dest_dir}/{f.stem}-{worker_id}{f.suffix}")
        moved += 1
    return moved, nbytes, time.monotonic() - t0


def count_pairs(store_dir: Path, sidecar_ext: str) -> int:
    """Pairs in a tag's store, by counting sidecars. A delivery interrupted
    between a pair's two members can leave an orphaned sidecar briefly counted
    here; it is inert to every consumer, so the count stays a progress reading
    rather than a completeness guarantee."""
    return sum(1 for _ in store_dir.glob(f"*{sidecar_ext}")) if store_dir.is_dir() else 0
