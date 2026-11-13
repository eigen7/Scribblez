"""The blind-spots viewer's opponent-rack panel, computed once per report
rather than by the viewer: a <stem>.beliefs.json sidecar beside each
.blindspots.json report holds, for each kept position, what the mover
believes about the opponent's rack under a leave inference model
(leave_inference/beliefs.py). Per tile, the chance the rack holds one under
the model and under no information, and SAMPLE_RACKS distinct sample racks,
the kept leave apart from the refill.

The hidden_blind_spots generator writes it under the model its pools were
labeled with. py/tools/one_off/backfill_blind_spots_beliefs.py writes it for
reports that predate it, possibly under a stand-in model; the sidecar names
its model, and the viewer flags one that differs from the report's.
"""

import json
import zlib
from pathlib import Path

import numpy as np

from scribblez.blind_spots_viewer import BELIEFS_SUFFIX, REPORT_SUFFIX, report_entries
from scribblez.leave_inference.beliefs import LeaveBeliefs, RackBelief, unseen_pool

SAMPLE_RACKS = 10
TILE_NAMES = [chr(ord("A") + i) for i in range(26)] + ["?"]


class BeliefWriter:
    """Writes sidecars from `model`, the export of leave inference tag
    `tag` at `generation`."""

    def __init__(self, model: LeaveBeliefs, tag: str, generation: int):
        self.model = model
        self.leave_model = {"tag": tag, "generation": generation}

    def write(self, report_path: Path, gcg_dir: Path) -> Path:
        """The sidecar of the report at `report_path`, whose kept positions'
        games are in `gcg_dir` and whose .slog and .forgone sit beside it."""
        stem = report_path.name.removesuffix(REPORT_SUFFIX)
        _, entries = report_entries(report_path, gcg_dir)
        positions = []
        if entries:
            beliefs = self.model.beliefs(
                report_path.with_name(f"{stem}.slog"),
                [(e["game"], e["turn"] - 1) for e in entries],
                [unseen_pool(e["board"], e["rack"]) for e in entries],
                [e["opp_rack_count"] for e in entries],
            )
            positions = [self._position(stem, e, b) for e, b in zip(entries, beliefs, strict=True)]
        out = report_path.with_name(stem + BELIEFS_SUFFIX)
        # Whole or absent, so a stopped backfill resumes past only finished ones.
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps({"leave_model": self.leave_model, "positions": positions}))
        tmp.replace(out)
        return out

    def _position(self, stem: str, entry: dict, b: RackBelief) -> dict:
        game, turn = entry["game"], entry["turn"] - 1
        # Seeded by the position, so a rewrite draws the same racks.
        rng = np.random.default_rng([zlib.crc32(stem.encode()), game, turn])
        inferred = self.model.rack_presence(b)
        uniform = self.model.rack_presence(b, uniform=True)
        return {
            "game": game,
            "turn": turn,
            "modelled": b.modelled,
            "leave_length": b.leave_length,
            "refill": b.refill,
            "presence": [
                {"tile": name, "inferred": float(i), "uniform": float(u)}
                for name, i, u in zip(TILE_NAMES, inferred, uniform, strict=True)
            ],
            "racks": self.model.sample_racks(b, SAMPLE_RACKS, rng),
        }
