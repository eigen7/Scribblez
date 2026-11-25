"""Controller-side measuring of a leaf_bot_generation tag's releases: spreading
their matches over the measure slots and recording what the slots deliver
(the RoleSpec.dispatch tick; measure.py is the worker's half).

It follows match eval's exchange (match_eval/dispatch.py): a slot's inbox is
the ledger of its assignments, an assignment stays until its result is
recorded, and ingest is idempotent. Each of a release's matches
(deep_bot/matches.py, against HastyBot and release 0) is an assignment of its
own, so that any number of slots share the releases due, and the controller
keeps a match off every slot but the one holding it. A slot plays one match at
a time; the newest release's come first, and older ones backfill. Every game
of each match is kept under the tag (matches.kept_dir), with the estimate the
chart draws in the database.

A match whose delivered result was quarantined (unreadable) has no row, so it
falls due again: its finished assignment is cleared like a recorded one's,
and the match goes to the next idle slot.
"""

from pathlib import Path

from scribblez.dashboard import db
from scribblez.deep_bot import matches, release
from scribblez.generational import lifecycle
from scribblez.leaf_bot_generation import LAYOUT, measure
from scribblez.paths import DONE_SUFFIX, TagPaths

# A match: (release, opponent).
Match = tuple[int, str]


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


def ingest(paths: TagPaths, conn) -> list[Match]:
    """Record every delivered match, oldest first: its estimate into the
    database, its games under the tag. Returns the matches recorded. A
    malformed result is quarantined (renamed to .bad), as match eval's is."""
    done = []
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
        done.append((record["release"], record["opponent"]))
    return done


def recorded(conn) -> set[Match]:
    """The matches whose result is in the database."""
    return {(row["release"], row["opponent"]) for row in db.read_release_matches(conn)}


def quarantined(paths: TagPaths) -> set[Match]:
    """The matches whose delivered result ingest quarantined."""
    found = set()
    for path in paths.match_results_dir.glob("release_*.bad"):
        found.add(measure.parse_token(path.with_suffix(".json").name))
    return found


def due(paths: TagPaths, done: set[Match], every: int) -> list[Match]:
    """The matches of the releases due a measurement that lack a result,
    the newest release's first."""
    newest = release.newest(paths)
    numbers = [] if newest is None else range(newest, -1, -1)
    return [
        (number, opponent)
        for number in numbers
        if number % every == 0 and release.exists(paths, number)
        for opponent in matches.opponents_of(number)
        if (number, opponent) not in done
    ]


def _inbox(paths: TagPaths, slot) -> str:
    return str(paths.match_inbox_dir(slot.worker_id).relative_to(paths.root))


def _assigned(paths: TagPaths, slot) -> list[str]:
    """The slot's assignments, played or not."""
    return slot.list(f"{_inbox(paths, slot)}/{measure.ASSIGNED_DIR}")


def _clear_spent(paths: TagPaths, slot, held: list[str], settled: set[Match]) -> list[str]:
    """Remove the slot's played assignments whose result is recorded or
    quarantined, with their release's files (release 0, an opponent, stays).
    Returns the assignments left."""
    inbox = _inbox(paths, slot)
    left = []
    for name in held:
        number, opponent = measure.parse_token(name)
        if not (name.endswith(DONE_SUFFIX) and (number, opponent) in settled):
            left.append(name)
            continue
        slot.remove(f"{inbox}/{measure.ASSIGNED_DIR}/{name}")
        for rel in LAYOUT.files(number) if number > 0 else []:
            slot.remove(f"{inbox}/{rel}")
    return left


def _put_release(paths: TagPaths, slot, number: int):
    """Give the slot release `number`, its manifest last."""
    inbox = _inbox(paths, slot)
    for rel in LAYOUT.files(number):
        slot.put(paths.root / rel, f"{inbox}/{rel}")


def _assign(paths: TagPaths, slot, match: Match):
    """Give an idle slot `match`: the releases it plays, then the
    assignment."""
    number, opponent = match
    inbox = _inbox(paths, slot)
    if release.MANIFEST not in slot.list(f"{inbox}/{release.release_rel(0)}"):
        _put_release(paths, slot, 0)
    if number > 0:
        _put_release(paths, slot, number)
    manifest = release.release_dir(paths, number) / release.MANIFEST
    slot.put(manifest, f"{inbox}/{measure.ASSIGNED_DIR}/{measure.token_name(number, opponent)}")


def assign(paths: TagPaths, conn, every: int, slots):
    """Give each idle slot the newest due match no slot holds. Lists the
    slots' inboxes, a round trip each to a remote slot, only when a match is
    due."""
    done = recorded(conn)
    pending = due(paths, done, every)
    if not pending:
        return
    settled = done | quarantined(paths)
    held = {
        slot.worker_id: _clear_spent(paths, slot, _assigned(paths, slot), settled) for slot in slots
    }
    taken = {measure.parse_token(name) for names in held.values() for name in names}
    free = [m for m in pending if m not in taken]
    for slot in slots:
        if not free:
            return
        if not held[slot.worker_id]:
            _assign(paths, slot, free.pop(0))


def outstanding(paths: TagPaths, conn, every: int) -> bool:
    """Whether a match may still be owed: a due release lacks one, or the
    releases here have not caught up with the trainer's cursor (a remote
    trainer's last release still on its way)."""
    if due(paths, recorded(conn), every):
        return True
    cursor = lifecycle.read_train_state(paths).get("generation_index", 0)
    newest = release.newest(paths)
    return newest is None or newest < cursor


def tick(spec, paths: TagPaths, params, slots) -> bool:
    """One controller-side pass for one task (the RoleSpec.dispatch hook).
    Returns whether a match may still be owed. The cheap filesystem checks
    come before opening the database, as in match eval's tick."""
    if not paths.dashboard_db.is_file():
        return True  # the trainer has not started
    if not slots and not any(paths.match_results_dir.glob("release_*.json")):
        return True
    conn = db.connect(paths.dashboard_db)
    try:
        ingest(paths, conn)
        if params.match_every_generations <= 0:
            return False  # measuring is off for this tag
        assign(paths, conn, params.match_every_generations, slots)
        return outstanding(paths, conn, params.match_every_generations)
    finally:
        conn.close()
