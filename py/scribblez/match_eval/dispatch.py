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

The inbox is the ledger, and it holds a generation until that generation is
accounted for here -- not until the worker is done with it. The worker marks
the model it has played (DONE_SUFFIX) rather than deleting it. The gap between
those two moments is real and wide: a container's result reaches the
controller by collection, which is a separate step of the reconcile pass and
can fail or time out for passes on end. A ledger that emptied when the worker
finished would offer that generation up again in exactly that gap, and the
machine would replay a match it had already played.

So an inbox holding either an unplayed export or a mark for an unaccounted
generation means that slot is spoken for. Ingest is idempotent anyway (a match
row is keyed by its generation), which is what keeps the database consistent
when a worker is interrupted and replays its own match.

Accounted for means recorded, or delivered and found unreadable: a quarantined
result is never coming back, so its mark stops holding the slot and the
generation simply falls due again and is replayed. Marks are cleared when the
slot is next offered work, not when the result lands -- there is nothing to
clear up for until then, and looking costs a round trip to another machine. A
mark can therefore outlive its row for as long as nothing new is due, which is
harmless: it blocks only assignment, and there is none to make.
"""

import json
from pathlib import Path

from scribblez.dashboard import db
from scribblez.paths import DONE_SUFFIX, ONNX_PREFIX, TagPaths

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
    "elapsed_s",
)


def _rel(paths: TagPaths, path: Path) -> str:
    """`path` as the slot-relative path both machines name it by."""
    return str(path.relative_to(paths.root))


def exported_generations(paths: TagPaths) -> list[int]:
    return sorted(paths.onnx_epoch(p) for p in paths.onnx_dir.glob(f"{ONNX_PREFIX}*.onnx"))


def recorded_generations(conn) -> set[int]:
    """Generations whose match is in the database."""
    return {r["epoch"] for r in conn.execute("SELECT epoch FROM match_eval")}


def pending_generation(paths: TagPaths, recorded: set[int], every: int) -> int | None:
    """The newest exported generation that is due a match and has none. Newest
    first keeps the readout tracking the training frontier; older stragglers
    backfill on later cycles."""
    pending = [g for g in exported_generations(paths) if g % every == 0 and g not in recorded]
    return max(pending) if pending else None


def _rows_trained_label(conn, gen: int) -> int:
    """The rows-clock the trainer recorded for this generation (the dashboard's
    alternate x-axis), 0 if the metrics row has not landed yet."""
    row = conn.execute(
        "SELECT value FROM metrics WHERE epoch = ? AND name = 'positions'", (gen,)
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def _delivered_results(paths: TagPaths) -> list[Path]:
    """Results a worker has delivered and nobody has recorded yet, oldest
    first."""
    return sorted(paths.match_results_dir.glob("*.json"))


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
    for path in _delivered_results(paths):
        try:
            ingested.append(_ingest_result(conn, path))
        except (AssertionError, ValueError) as e:
            print(f"match eval: quarantining unreadable result {path.name}: {e}")
            path.replace(path.with_suffix(".bad"))
            continue
        path.unlink()
    return ingested


def _inbox_epoch(name: str) -> int:
    """The generation an inbox entry stands for, played or not."""
    return TagPaths.onnx_epoch(Path(name.removesuffix(DONE_SUFFIX)))


def _result_epoch(path: Path) -> int:
    """The generation a delivered result's name carries (gen_NNNNNN-<worker>).
    Split at the first dash, since a worker id has dashes of its own."""
    return int(path.stem.split("-", 1)[0].removeprefix("gen_"))


def _quarantined_generations(paths: TagPaths) -> set[int]:
    """Generations whose delivered result was unreadable. Nothing further is
    coming for them, so they are as settled as recorded ones -- and, having no
    row, they fall due again and are replayed.

    A quarantined file whose name is not a result's is skipped: ingest sets
    aside whatever it finds in the directory, so this cannot assume the name
    was written by a worker of ours.
    """
    epochs = set()
    for path in paths.match_results_dir.glob("*.bad"):
        try:
            epochs.add(_result_epoch(path))
        except ValueError:
            continue
    return epochs


def _spent_marks(held: list[str], settled: set[int]) -> list[str]:
    """The marks in `held` whose generation is accounted for: their exchange is
    over, and the inbox entry is all that is left of it."""
    return [
        name
        for name in held
        if name.startswith(ONNX_PREFIX)
        and name.endswith(DONE_SUFFIX)
        and _inbox_epoch(name) in settled
    ]


def _assign(paths: TagPaths, conn, every: int, slot):
    """Give one idle slot the newest generation that is due a match.

    The local question is asked first because the other one is not local: a
    slot's listing is an ssh round trip for a container, and a tag whose
    matches are all played would otherwise pay one every pass to learn there
    was nothing to send.
    """
    recorded = recorded_generations(conn)
    gen = pending_generation(paths, recorded, every)
    if gen is None:
        return
    inbox = _rel(paths, paths.match_inbox_dir(slot.worker_id))
    held = slot.list(inbox)
    spent = _spent_marks(held, recorded | _quarantined_generations(paths))
    for name in spent:
        slot.remove(f"{inbox}/{name}")
    if any(name.startswith(ONNX_PREFIX) and name not in spent for name in held):
        return  # an unplayed export, or one whose result is still on its way
    # The shared external-data blobs go first and stay: a model does not load
    # without them, and they are identical for every generation.
    for sidecar in paths.onnx_sidecars:
        if sidecar.name not in held:
            slot.put(sidecar, f"{inbox}/{sidecar.name}")
    model = paths.onnx_path(gen)
    slot.put(model, f"{inbox}/{model.name}")


def tick(spec, tag: str, params, slots):
    """The RoleSpec.dispatch entry: one controller-side pass for one task.

    Every task of the workload is ticked on every reconcile pass, including
    long-finished ones, so the two questions that need no database -- has the
    trainer ever run, and is there anything to do -- are asked first. Opening
    the database is not free: it applies the schema and commits, which on an
    archived tag means recreating its write-ahead log every few seconds
    forever.
    """
    paths = spec.paths(tag)
    if not paths.dashboard_db.is_file():
        return  # the trainer has not started; there is nothing to match or record
    if not slots and not _delivered_results(paths):
        return  # no worker to assign to, and nothing waiting to be recorded
    conn = db.connect(paths.dashboard_db)
    try:
        ingest(paths, conn)
        if params.match_every_generations <= 0:
            return  # match eval is disabled for this tag
        for slot in slots:
            _assign(paths, conn, params.match_every_generations, slot)
    finally:
        conn.close()
