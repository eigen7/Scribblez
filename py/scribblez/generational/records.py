"""The trainer's record stream and the operator's controls file: how a train
role talks to the controller without touching dashboard.db.

A trainer emits immutable, generation-keyed records through its results sink,
and the controller's ingest tick (train_ingest.py) writes them into the tag's
dashboard.db -- the split match eval already uses for its results
(match_eval/dispatch.py), applied to training. The trainer is then the same
process wherever it runs: on this machine the sink is a rename into the tag
root, on a pod it is an upload; and the database has exactly one writer, the
dashboard server.

Records, under the tag's records/ directory (paths.py names them):

  run.json         the run's config: frozen params, parameter count, loss
                   weights, the controls' starting values. Rewritten freely
                   (a trainer re-stamps the parameter count once its model
                   exists).
  gen_NNNNNN.json  one trained generation: its metrics row, the control
                   events since the previous record, and which prediction
                   tables its .npz sidecar carries. A trainer writes it last
                   of the generation's outputs -- after the ONNX export, the
                   rolling checkpoint and train_state.json -- so it is also
                   the commit marker: its existence means everything the
                   generation produced is in place.
  gen_NNNNNN.npz   the per-dataset-position prediction arrays, keyed
                   "<table>/<array>" (dashboard/db.py's PRED_TABLES).

Controls travel the other way as one file, controls.json at the tag root:
the dashboard writes every control's current value there when the operator
sets one, and the trainer reads it at its natural cadence (once per
generation) through the same sink. A missing file or key means the trainer's
own default.

Torch-free, like everything a runner shares with the controller.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from scribblez.dashboard.db import PRED_TABLES
from scribblez.paths import (
    CONTROLS_REL,
    RUN_RECORD_REL,
    TagPaths,
    generation_preds_rel,
    generation_record_rel,
)


def _jsonable(value):
    """`value` with numpy scalars as Python numbers, so a metrics row built
    from tensor reductions serializes."""
    return value.item() if isinstance(value, np.generic) else value


class TrainRecorder:
    """The trainer's outbound side: what it has to say about the run and
    about each generation, delivered through its sink."""

    def __init__(self, sink):
        self._sink = sink
        self._events: list[dict] = []

    def publish_run(
        self, tag: str, params: dict, model_params: int, loss_weights: dict, controls: dict
    ):
        """The run's config (the Info tab, the Loss tab's stacking weights,
        the Controls tab's starting values). Idempotent; call again to
        re-stamp."""
        self._sink.push_json(
            RUN_RECORD_REL,
            {
                "tag": tag,
                "params": params,
                "model_params": int(model_params),
                "loss_weights": {k: float(v) for k, v in loss_weights.items()},
                "controls": {k: float(v) for k, v in controls.items()},
            },
        )

    def control_event(self, positions: int, name: str, value: float):
        """A control (or the LR schedule's phase) changed at `positions` rows
        trained. Held until the next generation's record carries it."""
        self._events.append(
            {"positions": int(positions), "name": name, "value": float(value), "t": time.time()}
        )

    def commit_generation(
        self, generation: int, positions: int, metrics: dict, preds: dict | None = None
    ):
        """Deliver a trained generation: `metrics` is its scalar row (the
        `positions` rows-clock included), `preds` maps a PRED_TABLES name to
        that table's arrays. Call once the generation's other outputs are in
        place -- this record is what says they are."""
        record = {
            "generation": int(generation),
            "positions": int(positions),
            "metrics": {k: _jsonable(v) for k, v in metrics.items()},
            "control_events": self._events,
            "preds": [],
        }
        if preds:
            self._push_preds(generation, preds)
            record["preds"] = sorted(preds)
        self._sink.push_json(generation_record_rel(generation), record)
        self._events = []

    def _push_preds(self, generation: int, preds: dict):
        arrays = {
            f"{table}/{name}": np.ascontiguousarray(preds[table][name])
            for table in preds
            for name in PRED_TABLES[table].arrays
        }
        fd, tmp = tempfile.mkstemp(suffix=".npz")
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.chmod(tmp, 0o644)  # mkstemp's private mode would follow the file into the tag
        self._sink.push_file(Path(tmp), generation_preds_rel(generation))


def read_controls(sink) -> dict[str, float]:
    """The operator's current controls, name -> value; empty until one has
    been set."""
    return sink.read_json(CONTROLS_REL) or {}


def write_controls_file(paths: TagPaths, controls: dict[str, float]):
    """Publish every control's current value for the trainer (the dashboard's
    side of the exchange), atomically."""
    path = paths.controls_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({k: float(v) for k, v in controls.items()}, indent=2) + "\n")
    os.replace(tmp, path)
