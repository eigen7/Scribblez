"""Controller-side match eval: assigning the matches and ingesting the results.

A match-eval worker does not choose its own work. It plays whatever export is
in its inbox and delivers the outcome as a file; this half -- ticked per task
by the dashboard's reconcile pass -- decides which generation that should be
and turns delivered outcomes into the dashboard.db rows the win-rate and LLR
curves read.

Splitting the role this way is what lets it run on another machine. The
database and the exports both live on the controller, neither of which an ssh
worker can reach; what crosses the link is one model in and one small JSON out
(cloud/ssh_transfer.py). A local slot takes the identical path -- its "link"
being a symlink into models/ -- so there is one runner and one set of rules
rather than one per kind.

The inbox is the ledger. A worker removes the model only after its result is
delivered, so an inbox still holding an export means that slot's match is
unfinished -- across a crash, a stopped container, a dashboard restart -- and
nothing is assigned over the top of it. Ingest is idempotent (a match row is
keyed by its generation), so a result that arrives twice costs nothing.
"""

import json
from pathlib import Path

from scribblez.dashboard import db
from scribblez.paths import ONNX_PREFIX, TagPaths

# What a delivered result must carry beyond the columns the controller fills
# in itself; a file missing any of them is not a match result.
RESULT_FIELDS = (
    "epoch",
    "opponent",
    "games",
    "wins",
    "draws",
    "losses",
    "pair_counts",
    "score",
    "ci_half_width",
    "llr",
    "llr_lower",
    "llr_upper",
    "decision",
    "elapsed_s",
)


def _rel(paths: TagPaths, path: Path) -> str:
    """`path` as the slot-relative path both machines name it by."""
    return str(path.relative_to(paths.root))


def exported_generations(paths: TagPaths) -> list[int]:
    return sorted(paths.onnx_epoch(p) for p in paths.onnx_dir.glob(f"{ONNX_PREFIX}*.onnx"))


def pending_generation(paths: TagPaths, conn, every: int) -> int | None:
    """The newest exported generation that is due a match and has none. Newest
    first keeps the readout tracking the training frontier; older stragglers
    backfill on later cycles."""
    done = {r["epoch"] for r in conn.execute("SELECT epoch FROM match_eval")}
    pending = [g for g in exported_generations(paths) if g % every == 0 and g not in done]
    return max(pending) if pending else None


def _rows_trained_label(conn, gen: int) -> int:
    """The rows-clock the trainer recorded for this generation (the dashboard's
    alternate x-axis), 0 if the metrics row has not landed yet."""
    row = conn.execute(
        "SELECT value FROM metrics WHERE epoch = ? AND name = 'positions'", (gen,)
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def _ingest_result(conn, path: Path) -> int:
    """Write one delivered result into the database. Returns its generation."""
    record = json.loads(path.read_text())
    missing = [f for f in RESULT_FIELDS if f not in record]
    assert not missing, f"{path.name}: result is missing {', '.join(missing)}"
    gen = int(record["epoch"])
    db.write_match_eval(conn, gen, {**record, "positions": _rows_trained_label(conn, gen)})
    db.write_metrics(
        conn,
        gen,
        {
            "match_score": record["score"],
            "match_llr": record["llr"],
            "match_games": record["games"],
        },
    )
    return gen


def ingest(paths: TagPaths, conn) -> list[int]:
    """Drain delivered results into the database, oldest first. Returns the
    generations ingested.

    A file that is not a result is set aside rather than retried forever: it
    can only have arrived whole (delivery is a rename, and a pull extracts
    atomically), so a malformed one is damage, and the tag's match readout must
    not stop advancing because of it.
    """
    ingested = []
    for path in sorted(paths.match_results_dir.glob("*.json")):
        try:
            ingested.append(_ingest_result(conn, path))
        except (AssertionError, ValueError) as e:
            print(f"match eval: quarantining unreadable result {path.name}: {e}")
            path.replace(path.with_suffix(".bad"))
            continue
        path.unlink()
    return ingested


def _assign(paths: TagPaths, conn, every: int, slot):
    """Give one idle slot the newest generation that is due a match."""
    inbox = _rel(paths, paths.match_inbox_dir(slot.worker_id))
    held = slot.list(inbox)
    if any(name.startswith(ONNX_PREFIX) for name in held):
        return  # that slot's match is still in flight
    gen = pending_generation(paths, conn, every)
    if gen is None:
        return
    # The shared external-data blobs go first and stay: a model does not load
    # without them, and they are identical for every generation.
    for sidecar in paths.onnx_sidecars:
        if sidecar.name not in held:
            slot.put(sidecar, f"{inbox}/{sidecar.name}")
    model = paths.onnx_path(gen)
    slot.put(model, f"{inbox}/{model.name}")
    print(f"match eval: assigned generation {gen} to {slot.worker_id}")


def tick(spec, tag: str, params, slots):
    """The RoleSpec.dispatch entry: one controller-side pass for one task."""
    paths = spec.paths(tag)
    if not paths.dashboard_db.is_file():
        return  # the trainer has not started; there is nothing to match or record
    conn = db.connect(paths.dashboard_db)
    try:
        ingest(paths, conn)
        if params.match_every_generations <= 0:
            return  # match eval is disabled for this tag
        for slot in slots:
            _assign(paths, conn, params.match_every_generations, slot)
    finally:
        conn.close()
