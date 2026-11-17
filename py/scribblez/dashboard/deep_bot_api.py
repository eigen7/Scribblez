"""Data for deep_bot's Positions tab: one known position at a time under one
of the tag's releases (deep_bot/position_view.py). Registered with the rest of
the data plane by api.make_app().

Routes (all but the first take task + tag):
  /api/deep_bot/positions   the known positions, in manifest order
  /api/deep_bot/releases    the tag's complete releases (the slider)
  /api/deep_bot/position    ?position=&release= -> the board, the release's
                            picks there, and its models' outputs

A release's files never change, so its models load once and each position's
analysis is memoized by (tag, release, position); the picks are read from the
tag's database every request, since the measure role records them after the
release exists. A position's analysis takes well under a second on the CPU.
"""

import functools
import subprocess
from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.web

from scribblez.dashboard import db
from scribblez.deep_bot import known_positions, position_view, release
from scribblez.deep_bot.position_view import Analysis, KnownPositionsReplay, ReleaseModels
from scribblez.paths import TagPaths
from scribblez.sim_survey_viewer import TILE_SCORES, bonuses

# The model work runs off the IOLoop on one thread, which also serializes
# access to the caches and the onnxruntime sessions.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@functools.cache
def _replay() -> KnownPositionsReplay:
    return KnownPositionsReplay()


# The caches key a tag by its (tag, task, mount_root): TagPaths hashes by
# identity, and every request makes its own.
@functools.lru_cache(maxsize=4)
def _models(tag_key: tuple[str, str, str], number: int) -> ReleaseModels:
    return ReleaseModels.load(TagPaths(*tag_key), number)


@functools.lru_cache(maxsize=256)
def _analysis(tag_key: tuple[str, str, str], number: int, index: int) -> Analysis:
    replay = _replay()
    view = position_view.board_view(replay.positions[index])
    return position_view.analyze(_models(tag_key, number), replay, index, view)


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
    p = _replay().positions[index]
    return {
        **position_view.payload(
            position_view.board_view(p),
            number,
            _picks(paths, number, p.id),
            _analysis(tag_key, number, index),
        ),
        "bonuses": bonuses(),
        "tile_scores": TILE_SCORES,
    }


class _Base(tornado.web.RequestHandler):
    @property
    def mount_root(self) -> str:
        return self.settings["mount_root"]

    def tag_key(self) -> tuple[str, str, str]:
        return self.get_query_argument("tag"), self.get_query_argument("task"), self.mount_root

    def tag_paths(self) -> TagPaths:
        return TagPaths(*self.tag_key())

    def fail(self, status: int, message: str):
        self.set_status(status)
        self.write({"error": message})


class PositionsHandler(_Base):
    async def get(self):
        replay = await tornado.ioloop.IOLoop.current().run_in_executor(_EXECUTOR, _replay)
        self.write({"positions": [{"id": p.id, "title": p.title} for p in replay.positions]})


class ReleasesHandler(_Base):
    def get(self):
        paths = self.tag_paths()
        picked = {r["release"] for r in _pick_rows(paths)}
        self.write({"releases": position_view.release_list(paths, picked)})


class PositionHandler(_Base):
    """One known position under one release; `release` defaults to the
    newest."""

    async def get(self):
        paths = self.tag_paths()
        numbers = release.numbers(paths)
        arg = self.get_query_argument("release", "latest")
        number = (numbers[-1] if numbers else None) if arg in ("", "latest") else int(arg)
        if number not in numbers:
            self.fail(404, "no such release" if numbers else "no release yet")
            return
        index = int(self.get_query_argument("position", "0"))
        if not 0 <= index < len(known_positions.load()):
            self.fail(404, "position out of range")
            return
        try:
            build = functools.partial(position_payload, self.tag_key(), number, index)
            self.write(await tornado.ioloop.IOLoop.current().run_in_executor(_EXECUTOR, build))
        except ValueError as e:  # a model the engine refuses, a belief it cannot summarize
            self.fail(400, str(e))
        except (OSError, subprocess.CalledProcessError) as e:  # engine unavailable
            self.fail(503, str(e))


ROUTES = [
    (r"/api/deep_bot/positions", PositionsHandler),
    (r"/api/deep_bot/releases", ReleasesHandler),
    (r"/api/deep_bot/position", PositionHandler),
]
