"""Controller-side match eval: assigning matches and ingesting results.

A match-eval worker (runner.py) plays whatever export is in its inbox and
delivers the outcome as a file. This half, ticked per task by the dashboard's
reconcile pass, decides which generation each worker plays and turns delivered
outcomes into the dashboard.db rows behind the match win-rate curve.

The split is what lets the worker run on another machine: the database and
the exports live on the controller, and only one model in and one small JSON
out cross the link (cloud/ssh_transfer.py). A local slot takes the same path,
with a symlink into models/ as its "link", so there is one set of rules.

The inbox is the ledger of assignments. A generation stays in it until the
controller has accounted for the result, not merely until the worker has
played it: the worker marks a played model with DONE_SUFFIX rather than
deleting it. For a remote worker the result arrives by a separate collection
step, which can lag or fail for many passes, and an inbox that emptied on
completion would get the same generation assigned and replayed in that gap.
So a slot is busy while its inbox holds an unplayed export or a mark for an
unaccounted generation. Ingest is idempotent (rows are keyed by generation),
so an interrupted worker replaying its own match is harmless.

A generation is accounted for once its result is recorded, or once its
delivered result was found unreadable and quarantined; the latter has no row,
so it falls due again and is replayed. Marks are cleared lazily, when the slot
is next offered work, since checking a remote inbox costs a round trip. A mark
can thus outlive its purpose while nothing new is due, which is harmless: it
blocks only assignment.
"""

import json
from pathlib import Path

from scribblez.dashboard import db
from scribblez.generational import lifecycle
from scribblez.paths import DONE_SUFFIX, ONNX_PREFIX, TagPaths

# Fields a delivered result must carry (the controller adds the rest itself).
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
    """`path` relative to the tag root, which is how both machines name it."""
    return str(path.relative_to(paths.root))


def recorded_generations(conn) -> set[int]:
    """Generations whose match is in the database."""
    return {r["epoch"] for r in conn.execute("SELECT epoch FROM match_eval")}


def pending_generation(paths: TagPaths, recorded: set[int], every: int) -> int | None:
    """The newest exported generation that is due a match and has none. Newest
    first keeps the readout at the training frontier; older ones backfill
    later."""
    pending = [g for g in paths.exported_generations() if g % every == 0 and g not in recorded]
    return max(pending) if pending else None


def _rows_trained_label(conn, gen: int) -> int:
    """Rows trained as of generation `gen` (the dashboard's alternate x-axis),
    or 0 if its metrics row has not been ingested yet."""
    row = conn.execute(
        "SELECT value FROM metrics WHERE epoch = ? AND name = 'positions'", (gen,)
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def _delivered_results(paths: TagPaths) -> list[Path]:
    """Delivered results not yet ingested, oldest first."""
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
    """Ingest delivered results into the database, oldest first. Returns the
    generations ingested.

    A malformed file is quarantined (renamed to .bad) rather than retried.
    Delivery is atomic, so it is not a partial write that might yet complete,
    and it must not stall the tag's match readout.
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
    """The generation in a result filename, gen_NNNNNN-<worker_id> (worker ids
    may contain dashes, so split at the first)."""
    return int(path.stem.split("-", 1)[0].removeprefix("gen_"))


def _quarantined_generations(paths: TagPaths) -> set[int]:
    """Generations whose delivered result was quarantined. Ingest quarantines
    any file in the directory, so names that do not parse are skipped."""
    epochs = set()
    for path in paths.match_results_dir.glob("*.bad"):
        try:
            epochs.add(_result_epoch(path))
        except ValueError:
            continue
    return epochs


def _spent_marks(held: list[str], settled: set[int]) -> list[str]:
    """The done-marks in `held` whose generation is accounted for."""
    return [
        name
        for name in held
        if name.startswith(ONNX_PREFIX)
        and name.endswith(DONE_SUFFIX)
        and _inbox_epoch(name) in settled
    ]


def _assign(paths: TagPaths, conn, every: int, slot):
    """Give one slot the newest generation due a match, if the slot is idle.

    Checks for due work before listing the slot's inbox, because listing a
    remote inbox costs an ssh round trip and most passes have nothing to assign.
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
        # The worker marks only a move-proposal export's cache graph; clear its
        # step/ companion too.
        slot.remove(f"{inbox}/step/{name.removesuffix(DONE_SUFFIX)}")
    if any(name.startswith(ONNX_PREFIX) and name not in spent for name in held):
        return  # busy: an unplayed export, or a result still in transit
    # Sidecars go first and stay: models need them to load, and they are shared
    # by every generation.
    for sidecar in paths.onnx_sidecars:
        if sidecar.name not in held:
            slot.put(sidecar, f"{inbox}/{sidecar.name}")
    # A move-proposal export is a pair. The worker polls for the cache graph,
    # so the step graph must arrive first.
    model = paths.onnx_path(gen)
    step = paths.proposal_step_path(gen)
    if step.exists():
        slot.put(step, f"{inbox}/step/{step.name}")
    slot.put(model, f"{inbox}/{model.name}")


def tick(spec, tag: str, params, slots) -> bool:
    """One controller-side pass for one task (the RoleSpec.dispatch hook).
    Returns whether match work may still be outstanding (see _outstanding); the
    dashboard finishes the role once it is not and the trainer has finished.

    Every task is ticked on every reconcile pass, including long-finished ones,
    so the cheap filesystem checks come before opening the database. Opening it
    applies the schema and commits, which on an archived tag would recreate its
    write-ahead log every few seconds.
    """
    paths = spec.paths(tag)
    if not paths.dashboard_db.is_file():
        return True  # the trainer has not started; there is nothing to match yet
    if not slots and not _delivered_results(paths):
        return True  # nothing to assign or record; not worth opening the database
    conn = db.connect(paths.dashboard_db)
    try:
        ingest(paths, conn)
        if params.match_every_generations <= 0:
            return False  # match eval is disabled for this tag
        for slot in slots:
            _assign(paths, conn, params.match_every_generations, slot)
        return _outstanding(paths, conn, params.match_every_generations)
    finally:
        conn.close()


def _outstanding(paths: TagPaths, conn, every: int) -> bool:
    """Whether a match may still be owed: a due generation has no recorded
    result (it is unassigned, being played, or its result is in transit), or
    the exports have not caught up with the trainer's cursor. The second case
    covers a remote trainer whose final export is still on its way to the
    controller when the trainer is already seen to have finished."""
    if pending_generation(paths, recorded_generations(conn), every) is not None:
        return True
    cursor = lifecycle.read_train_state(paths).get("generation_index", 0)
    exported = paths.exported_generations()
    return cursor > 0 and (not exported or exported[-1] < cursor - 1)
