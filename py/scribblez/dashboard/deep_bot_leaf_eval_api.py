"""Data for deep_bot_leaf_eval's Positions tab: one known position at a time
under one generation's analysis (deep_bot_leaf_eval/positions.py), which the
match slot delivered after that generation's match. Registered with the rest
of the data plane by api.make_app(). The known positions themselves come from
/api/deep_bot/positions.

Routes (both take task + tag):
  /api/deep_bot_leaf_eval/generations  the analyzed generations (the slider),
                                       with their rows trained and match score
  /api/deep_bot_leaf_eval/position     ?position=&generation= -> the board,
                                       the matches' pick there, and the leaf's
                                       value of the moves worth listing

A delivered analysis never changes, so each is read once.
"""

import functools
import json

import tornado.web

from scribblez.dashboard import db
from scribblez.deep_bot import known_positions
from scribblez.deep_bot.position_view import board_view, move_json
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.paths import TagPaths
from scribblez.sim_survey_viewer import TILE_SCORES, bonuses

# The likeliest moves under the leaf the table lists, besides any that the
# reference, static equity or the pick names.
SHOWN_VALUES = 25


def analyzed_generations(paths: TagPaths) -> list[int]:
    """The generations with a delivered analysis, ascending."""
    folder = paths.data_dir / leaf_positions.POSITIONS_DIR
    return sorted(leaf_positions.generation_of(f) for f in folder.glob("gen_*.json"))


# Keyed by the tag's (tag, task, mount_root): TagPaths hashes by identity, and
# every request makes its own. A missing file raises, which is not cached.
@functools.lru_cache(maxsize=32)
def _analysis(tag_key: tuple[str, str, str], generation: int) -> dict:
    path = TagPaths(*tag_key).data_dir / leaf_positions.rel(generation)
    return json.loads(path.read_text())


def _matches(paths: TagPaths) -> dict[int, dict]:
    """The tag's match results by generation; none before its first."""
    if not paths.dashboard_db.is_file():
        return {}
    conn = db.connect(paths.dashboard_db)
    try:
        return {r["epoch"]: r for r in db.read_all_match_eval(conn)}
    finally:
        conn.close()


def generation_list(paths: TagPaths) -> list[dict]:
    """The slider's entries: each analyzed generation, its rows trained (from
    its match row) and its match score, when the match is recorded."""
    matches = _matches(paths)
    out = []
    for g in analyzed_generations(paths):
        m = matches.get(g)
        out.append(
            {
                "generation": g,
                "rows_trained": m["positions"] if m else None,
                "score": m["score"] if m else None,
                "ci_half_width": m["ci_half_width"] if m else None,
            }
        )
    return out


def values_json(values: list[dict], named: set[str], board) -> list[dict]:
    """The value table's rows, best first: the SHOWN_VALUES best under the
    leaf and any move in `named`, each drawn on the pre-move `board`."""
    return [
        {**v, **move_json(v["move"], board)}
        for v in values
        if v["leaf_rank"] <= SHOWN_VALUES or v["move"] in named
    ]


def payload(view: dict, generation: int, analysis: dict, match: dict | None) -> dict:
    """The tab's view of a known position under one generation: its
    board_view, the matches' pick there and how it reads against the answer,
    and the leaf's values of the moves worth listing."""
    entry = analysis["positions"][view["id"]]
    board = view["board"]
    named = {m["move"] for m in [*view["reference"], view["static_equity_best"]]} | {entry["pick"]}
    return {
        **view,
        "generation": generation,
        "candidates": analysis["candidates"],
        "pick": {**move_json(entry["pick"], board), "reading": entry["reading"]},
        "num_values": len(entry["values"]),
        "values": values_json(entry["values"], named, board),
        "match": None
        if match is None
        else {k: match[k] for k in ("score", "ci_half_width", "games", "opponent")},
    }


def position_payload(tag_key: tuple[str, str, str], generation: int, index: int) -> dict:
    paths = TagPaths(*tag_key)
    p = known_positions.load()[index]
    return {
        **payload(
            board_view(p),
            generation,
            _analysis(tag_key, generation),
            _matches(paths).get(generation),
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


class GenerationsHandler(_Base):
    def get(self):
        self.write({"generations": generation_list(TagPaths(*self.tag_key()))})


class PositionHandler(_Base):
    """One known position under one analyzed generation; `generation`
    defaults to the newest."""

    def get(self):
        analyzed = analyzed_generations(TagPaths(*self.tag_key()))
        arg = self.get_query_argument("generation", "latest")
        generation = (analyzed[-1] if analyzed else None) if arg in ("", "latest") else int(arg)
        if generation not in analyzed:
            self.set_status(404)
            self.write({"error": "no such analysis" if analyzed else "no analysis yet"})
            return
        index = int(self.get_query_argument("position", "0"))
        if not 0 <= index < len(known_positions.load()):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        self.write(position_payload(self.tag_key(), generation, index))


ROUTES = [
    (r"/api/deep_bot_leaf_eval/generations", GenerationsHandler),
    (r"/api/deep_bot_leaf_eval/position", PositionHandler),
]
