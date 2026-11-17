"""Data for deep_bot's Positions tab: one known position at a time under one
of the tag's releases (deep_bot/position_view.py). Registered with the rest of
the data plane by api.make_app().

Routes (all but the first take task + tag):
  /api/deep_bot/positions   the known positions, in manifest order
  /api/deep_bot/releases    the tag's complete releases (the slider)
  /api/deep_bot/position    ?position=&release= -> the board, the release's
                            picks there, and its models' outputs

The models' outputs are the release's positions.json, which the trainer wrote
as it published the release (deep_bot/release_positions.py); a release's files
never change, so each is read once. The picks are read from the tag's
database every request, since the measure role records them after the release
exists.
"""

import functools
import json

import tornado.web

from scribblez.dashboard import db
from scribblez.deep_bot import known_positions, position_view, release, release_positions
from scribblez.paths import TagPaths
from scribblez.sim_survey_viewer import TILE_SCORES, bonuses


# Keyed by the tag's (tag, task, mount_root): TagPaths hashes by identity, and
# every request makes its own. A missing file raises, which is not cached.
@functools.lru_cache(maxsize=16)
def _analysis_file(tag_key: tuple[str, str, str], number: int) -> dict:
    return json.loads(release_positions.path(TagPaths(*tag_key), number).read_text())


def _analysis(tag_key: tuple[str, str, str], number: int, position_id: str) -> dict | None:
    """Release `number`'s analysis of the position, as position_view.payload
    takes it; None for a release published before the trainer wrote one."""
    try:
        doc = _analysis_file(tag_key, number)
    except FileNotFoundError:
        return None
    entry = doc["positions"].get(position_id)
    return None if entry is None else {**entry, "inferred": doc["inferred"]}


def _pick_rows(paths: TagPaths) -> list[dict]:
    """Every recorded pick of the tag; none before the measure role has
    recorded any."""
    if not paths.dashboard_db.is_file():
        return []
    conn = db.connect(paths.dashboard_db)
    try:
        return db.read_release_picks(conn)
    finally:
        conn.close()


def _picks(paths: TagPaths, number: int, position_id: str) -> dict[str, dict]:
    """Release `number`'s picks at the position, by reply model."""
    return {
        r["reply"]: r
        for r in _pick_rows(paths)
        if r["release"] == number and r["position"] == position_id
    }


def position_payload(tag_key: tuple[str, str, str], number: int, index: int) -> dict:
    paths = TagPaths(*tag_key)
    p = known_positions.load()[index]
    return {
        **position_view.payload(
            position_view.board_view(p),
            number,
            _picks(paths, number, p.id),
            _analysis(tag_key, number, p.id),
        ),
        "bonuses": bonuses(),
        "tile_scores": TILE_SCORES,
    }


class _Base(tornado.web.RequestHandler):
    def tag_key(self) -> tuple[str, str, str]:
        return (
            self.get_query_argument("tag"),
            self.get_query_argument("task"),
            self.settings["mount_root"],
        )

    def tag_paths(self) -> TagPaths:
        return TagPaths(*self.tag_key())


class PositionsHandler(_Base):
    def get(self):
        positions = known_positions.load()
        self.write({"positions": [{"id": p.id, "title": p.title} for p in positions]})


class ReleasesHandler(_Base):
    def get(self):
        paths = self.tag_paths()
        picked = {r["release"] for r in _pick_rows(paths)}
        self.write({"releases": position_view.release_list(paths, picked)})


class PositionHandler(_Base):
    """One known position under one release; `release` defaults to the
    newest."""

    def get(self):
        numbers = release.numbers(self.tag_paths())
        arg = self.get_query_argument("release", "latest")
        number = (numbers[-1] if numbers else None) if arg in ("", "latest") else int(arg)
        if number not in numbers:
            self.set_status(404)
            self.write({"error": "no such release" if numbers else "no release yet"})
            return
        index = int(self.get_query_argument("position", "0"))
        if not 0 <= index < len(known_positions.load()):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        self.write(position_payload(self.tag_key(), number, index))


ROUTES = [
    (r"/api/deep_bot/positions", PositionsHandler),
    (r"/api/deep_bot/releases", ReleasesHandler),
    (r"/api/deep_bot/position", PositionHandler),
]
