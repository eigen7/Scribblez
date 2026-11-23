"""Controller-side measuring of a deep_bot tag's releases: assigning each to
the measure slot and recording the matches it delivers (the RoleSpec.dispatch
tick; measure.py is the worker's half).

It follows match eval's exchange (match_eval/dispatch.py), whose reasoning it
does not repeat: the slot's inbox is the ledger of assignments, an assignment
stays until its results are recorded, and ingest is idempotent. What differs
is what is assigned, a release with release 0 and the known positions beside
it in place of one model, and what comes back: the release's picks at those
positions, and every game of each match, kept under the tag
(matches.kept_dir), with the estimate the chart draws in the database.
"""

import json
from pathlib import Path

from scribblez.dashboard import db
from scribblez.deep_bot import known_positions, matches, measure, release
from scribblez.generational import lifecycle
from scribblez.paths import DONE_SUFFIX, TagPaths


def _rel(paths: TagPaths, path: Path) -> str:
    return str(path.relative_to(paths.root))


def measured(conn) -> set[int]:
    """The releases whose picks and every one of whose matches are recorded."""
    played: dict[int, set[str]] = {}
    for row in db.read_release_matches(conn):
        played.setdefault(row["release"], set()).add(row["opponent"])
    asked = {row["release"] for row in db.read_release_picks(conn)}
    return {r for r in asked if played.get(r, set()) >= set(matches.opponents_of(r))}


def pending_release(paths: TagPaths, done: set[int], every: int) -> int | None:
    """The newest release due its matches that lacks one. Newest first keeps
    the readout at the frontier; older ones backfill."""
    newest = release.newest(paths)
    due = [] if newest is None else [r for r in range(newest + 1) if r % every == 0]
    pending = [r for r in due if r not in done and release.exists(paths, r)]
    return max(pending, default=None)


def _keep(paths: TagPaths, record: dict, delivered: Path):
    """Move a delivered match's rows, and its games if they have arrived,
    under the release's kept matches."""
    number, opponent = record["release"], record["opponent"]
    kept = matches.kept_dir(paths, number)
    kept.mkdir(parents=True, exist_ok=True)
    games = paths.data_dir / matches.games_rel(number, opponent)
    if games.exists():
        games.replace(kept / f"{opponent}.slog")
    delivered.replace(kept / f"{opponent}.json")


def ingest_picks(paths: TagPaths, conn) -> list[int]:
    """Record every delivered set of picks and keep it under its release.
    Returns the releases recorded; a malformed file is quarantined."""
    recorded = []
    for path in sorted(paths.match_results_dir.glob("picks_release_*.json")):
        try:
            record = json.loads(path.read_text())
            db.write_release_picks(conn, int(record["release"]), record["picks"])
        except (ValueError, KeyError, TypeError) as e:
            print(f"measure: quarantining unreadable picks {path.name}: {e}")
            path.replace(path.with_suffix(".bad"))
            continue
        kept = matches.kept_dir(paths, int(record["release"]))
        kept.mkdir(parents=True, exist_ok=True)
        path.replace(kept / "picks.json")
        recorded.append(int(record["release"]))
    return recorded


def ingest(paths: TagPaths, conn) -> list[tuple[int, str]]:
    """Record every delivered match, oldest first: its estimate into the
    database, its games under the tag. Returns the (release, opponent)
    recorded. A malformed result is quarantined (renamed to .bad), as match
    eval's is."""
    recorded = []
    for path in sorted(paths.match_results_dir.glob("release_*.json")):
        try:
            record = matches.read_result(path)
            summary = matches.estimate(record["games"])
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"measure: quarantining unreadable result {path.name}: {e}")
            path.replace(path.with_suffix(".bad"))
            continue
        summary["elapsed_s"] = record["elapsed_s"]
        db.write_release_match(conn, record["release"], record["opponent"], summary)
        _keep(paths, record, path)
        recorded.append((record["release"], record["opponent"]))
    return recorded


def _token_release(name: str) -> int:
    return int(name.removesuffix(DONE_SUFFIX).removesuffix(".json").removeprefix("release_"))


def _clear_spent(slot, inbox: str, held: list[str], done: set[int]) -> list[str]:
    """Remove the finished assignments in `held` whose release is recorded,
    with that release's files (release 0, the opponent, stays). Returns the
    assignments left."""
    left = []
    for name in held:
        number = _token_release(name)
        if not (name.endswith(DONE_SUFFIX) and number in done):
            left.append(name)
            continue
        slot.remove(f"{inbox}/{measure.ASSIGNED_DIR}/{name}")
        for rel in release.DEEP_BOT.files(number) if number > 0 else []:
            slot.remove(f"{inbox}/{rel}")
    return left


def _put_known_positions(slot, inbox: str):
    """Give the slot the known positions, once: the manifest last."""
    held = f"{inbox}/{matches.KNOWN_POSITIONS_DIR}"
    files = known_positions.files()
    if files[-1].name in slot.list(held):
        return
    for path in files:
        slot.put(path, f"{held}/{path.name}")


def _put_release(paths: TagPaths, slot, inbox: str, number: int):
    """Give the slot release `number`, its manifest last."""
    for rel in release.DEEP_BOT.files(number):
        slot.put(paths.root / rel, f"{inbox}/{rel}")


def _assign(paths: TagPaths, conn, every: int, slot):
    """Give one slot the newest release due its matches, if the slot is idle."""
    done = measured(conn)
    number = pending_release(paths, done, every)
    if number is None:
        return
    inbox = _rel(paths, paths.match_inbox_dir(slot.worker_id))
    held = slot.list(f"{inbox}/{measure.ASSIGNED_DIR}")
    if _clear_spent(slot, inbox, held, done):
        return  # busy: an assignment unplayed, or its results still in transit
    # The positions and the opponent first; they stay for every assignment.
    _put_known_positions(slot, inbox)
    if release.MANIFEST not in slot.list(f"{inbox}/{Path(release.DEEP_BOT.files(0)[-1]).parent}"):
        _put_release(paths, slot, inbox, 0)
    if number > 0:
        _put_release(paths, slot, inbox, number)
    token = f"release_{number:06d}.json"
    manifest = release.release_dir(paths, number) / release.MANIFEST
    slot.put(manifest, f"{inbox}/{measure.ASSIGNED_DIR}/{token}")


def tick(spec, paths: TagPaths, params, slots) -> bool:
    """One controller-side pass for one task (the RoleSpec.dispatch hook).
    Returns whether a match may still be owed. The cheap filesystem checks
    come before opening the database, as in match eval's tick."""
    if not paths.dashboard_db.is_file():
        return True  # the trainer has not started
    if not slots and not any(paths.match_results_dir.glob("*release_*.json")):
        return True
    conn = db.connect(paths.dashboard_db)
    try:
        ingest_picks(paths, conn)
        ingest(paths, conn)
        if params.match_every_generations <= 0:
            return False  # measuring is off for this tag
        for slot in slots:
            _assign(paths, conn, params.match_every_generations, slot)
        return _outstanding(paths, conn, params.match_every_generations)
    finally:
        conn.close()


def _outstanding(paths: TagPaths, conn, every: int) -> bool:
    """Whether a match may still be owed: a due release lacks one, or the
    releases here have not caught up with the trainer's cursor (a remote
    trainer's last release still on its way)."""
    if pending_release(paths, measured(conn), every) is not None:
        return True
    cursor = lifecycle.read_train_state(paths).get("generation_index", 0)
    newest = release.newest(paths)
    return newest is None or newest < cursor
