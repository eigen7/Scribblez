"""Controller-side ingest of a trainer's records into dashboard.db.

The other half of generational/records.py. The dashboard's reconcile pass
ticks this per task for every role that declares it (RoleSpec.ingest). The run
record becomes the meta row, loss weights and the controls' starting values;
each generation record becomes a metrics row, control events and
prediction-table rows. The trainer never opens the database, so this is the
one place its results become rows, wherever it runs.

Every task is ticked on every pass, long-finished ones included, so a pass
must be cheap when nothing is new. The records directory's mtime is compared
with the last pass's before the database is opened, because opening applies
the schema and commits, which on an archived tag would recreate its
write-ahead log every few seconds forever. Each record is ingested once per
file identity (name, mtime, size), tracked in a ledger table in the database:
a rewritten run.json is ingested again, an unchanged generation record never
is.

Records are ingested oldest generation first, so metrics rows appear in
training order. A record that cannot be read is skipped and retried on later
passes. The sink writes records atomically, so an unreadable one is not torn;
it means a trainer running newer code than this controller, which a redeploy
fixes. The reverse skew, an older trainer still delivering a prediction table
this controller has dropped, is not an error: the table is left unread.
"""

import json
import re
from pathlib import Path

import numpy as np

from scribblez.dashboard import db
from scribblez.generational.records import write_controls_file
from scribblez.paths import TagPaths

_GENERATION_RECORD = re.compile(r"gen_(\d{6})\.json$")

# records/ directory mtime as of the last pass, per tag root. A dashboard
# restart forgets it and pays one database open per tag.
_seen_dir_mtime: dict[Path, int] = {}


def _file_identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _pending(paths: TagPaths, ledger: dict) -> list[Path]:
    """Records not yet ingested at their current identity: run.json first,
    then generations in order."""
    pending = []
    run = paths.run_record_path
    if run.is_file() and ledger.get(run.name) != _file_identity(run):
        pending.append(run)
    gens = sorted(
        (int(m.group(1)), p)
        for p in paths.records_dir.glob("gen_*.json")
        if (m := _GENERATION_RECORD.search(p.name))
    )
    pending.extend(p for _, p in gens if ledger.get(p.name) != _file_identity(p))
    return pending


def _ingest_run(conn, record: dict):
    db.write_meta(conn, record["tag"], record["params"], record["model_params"])
    db.write_loss_weights(conn, record["loss_weights"])
    # The operator's own values, if any, win over the trainer's defaults.
    db.init_control(conn, record["controls"])


def _ingest_generation(conn, paths: TagPaths, record: dict):
    gen, positions = record["generation"], record["positions"]
    db.write_metrics(conn, gen, record["metrics"])
    for e in record["control_events"]:
        db.write_control_event(conn, e["positions"], e["name"], e["value"], t=e["t"])
    tables = [t for t in record["preds"] if t in db.PRED_TABLES]  # retired ones left unread
    if tables:
        with np.load(paths.records_dir / f"gen_{gen:06d}.npz") as npz:
            for table in tables:
                arrays = {name: npz[f"{table}/{name}"] for name in db.PRED_TABLES[table].arrays}
                db.PRED_TABLES[table].write(conn, gen, positions, arrays)


def ingest(paths: TagPaths, conn) -> list[str]:
    """Write every record not yet ingested, oldest first. Returns the names
    written."""
    written = []
    for path in _pending(paths, db.read_record_ledger(conn)):
        identity = _file_identity(path)
        try:
            record = json.loads(path.read_text())
            if path.name == paths.run_record_path.name:
                _ingest_run(conn, record)
            else:
                _ingest_generation(conn, paths, record)
        except (KeyError, ValueError, OSError) as e:
            print(f"train ingest: skipping {path.name}: {e}")
            continue
        db.write_record_ledger(conn, path.name, *identity)
        written.append(path.name)
    return written


def _controls_file_current(paths: TagPaths, conn):
    """Write the controls file from the database when it is missing. A tag
    created before the file existed has the operator's values only in the
    database; this keeps its trainer running under them, not its defaults."""
    if not paths.controls_path.exists():
        write_controls_file(paths, db.read_controls(conn))


def tick(spec, tag: str):
    """The RoleSpec.ingest entry: one controller-side pass for one task."""
    paths = spec.paths(tag)
    records = paths.records_dir
    if not records.is_dir():
        return  # the trainer has never run
    mtime = records.stat().st_mtime_ns
    if _seen_dir_mtime.get(paths.root) == mtime:
        return  # nothing delivered since the last pass
    conn = db.connect(paths.dashboard_db)
    try:
        _controls_file_current(paths, conn)
        ingest(paths, conn)
    finally:
        conn.close()
    _seen_dir_mtime[paths.root] = mtime


def forget(paths: TagPaths):
    """Drop the pass cache for a tag (tests, and a tag deleted then recreated)."""
    _seen_dir_mtime.pop(paths.root, None)
