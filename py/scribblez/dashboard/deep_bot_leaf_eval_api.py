"""Data for deep_bot_leaf_eval's Positions tab: one known position at a time
under one generation's analysis (deep_bot_leaf_eval/positions.py), which the
match slot delivered after that generation's match. Registered with the rest
of the data plane by api.make_app(). The known positions themselves come from
/api/deep_bot/positions.

Routes (all take task + tag):
  /api/deep_bot_leaf_eval/generations  the analyzed generations (the slider),
                                       with their rows trained and match score
  /api/deep_bot_leaf_eval/position     ?position=&generation= -> the board,
                                       the matches' pick there, and the leaf's
                                       value of the moves worth listing, beside
                                       the truth's (deep_bot_leaf_eval/truth.py)
                                       where the repo holds one
  /api/deep_bot_leaf_eval/history      ?position=&move= -> the leaf's value of
                                       one move under every analyzed generation

A delivered analysis never changes, so each is read once.
"""

import functools
import json
from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.web

from scribblez import workloads
from scribblez.dashboard import db, tasks
from scribblez.deep_bot import known_positions
from scribblez.deep_bot.position_view import board_view, move_json
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.deep_bot_leaf_eval import truth
from scribblez.paths import TagPaths
from scribblez.sim_survey_viewer import TILE_SCORES, bonuses

# The likeliest moves under the leaf the table lists, besides any that the
# reference, static equity or the pick names.
SHOWN_VALUES = 25
# The truth's best moves the table lists too.
SHOWN_TRUTH = 10
# A history's first request for a position reads every analysis, so it runs
# off the IOLoop; one thread, as the caches are shared.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


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


# One position's values out of one generation's analysis, kept apart from
# _analysis so a history's sweep over every generation does not evict the
# slider's. A position's values over 80 generations are ~12 MB.
@functools.lru_cache(maxsize=1024)
def _position_values(tag_key: tuple[str, str, str], generation: int, position: str) -> dict:
    path = TagPaths(*tag_key).data_dir / leaf_positions.rel(generation)
    entry = json.loads(path.read_text())["positions"][position]
    return {v["move"]: v["value"] for v in entry["values"]}


def history(tag_key: tuple[str, str, str], position: str, move: str) -> list[dict]:
    """The leaf's value of `move` at `position` under each analyzed
    generation, ascending; null under one that did not value it."""
    return [
        {"generation": g, "value": _position_values(tag_key, g, position).get(move)}
        for g in analyzed_generations(TagPaths(*tag_key))
    ]


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


def truth_ranks(values: list[dict], truths: dict[str, float]) -> dict[str, int]:
    """Each move's rank under the truth `truths`, the better static equity
    first on ties, over the leaf's `values` the truth covers."""
    covered = sorted(
        (v for v in values if v["move"] in truths),
        key=lambda v: (-truths[v["move"]], v["equity_rank"]),
    )
    return {v["move"]: i + 1 for i, v in enumerate(covered)}


def values_json(values: list[dict], named: set[str], board, truths: dict[str, float]) -> list[dict]:
    """The value table's rows, best first: the SHOWN_VALUES best under the
    leaf, the SHOWN_TRUTH best under the truth and any move in `named`, each
    drawn on the pre-move `board` and with its truth (null where it has none)."""
    ranks = truth_ranks(values, truths)
    return [
        {
            **v,
            **move_json(v["move"], board),
            "truth": truths.get(v["move"]),
            "truth_rank": ranks.get(v["move"]),
        }
        for v in values
        if v["leaf_rank"] <= SHOWN_VALUES
        or ranks.get(v["move"], SHOWN_TRUTH + 1) <= SHOWN_TRUTH
        or v["move"] in named
    ]


def truth_json(doc: dict | None, truths: dict | None, leaf_params: dict | None) -> dict | None:
    """What the table says about the position's truth, from the repo's file
    `doc`: whose it is and how it was run, whether its opponent leaves are the
    tag's analysis's, and how many moves it covers; None without one."""
    if truths is None:
        return None
    return {
        "tag": doc["tag"],
        "run": doc["run"],
        "same_leaves": truth.same_leaves(doc["run"], leaf_params or {}),
        "num_values": len(truths),
    }


def payload(
    view: dict,
    generation: int,
    analysis: dict,
    match: dict | None,
    truths: dict[str, float] | None = None,
    truth_about: dict | None = None,
) -> dict:
    """The tab's view of a known position under one generation: its
    board_view, the matches' pick there and how it reads against the answer,
    and the leaf's values of the moves worth listing beside their truth
    `truths` (truth.of_position), which `truth_about` (truth_json) describes."""
    entry = analysis["positions"][view["id"]]
    board = view["board"]
    named = {m["move"] for m in [*view["reference"], view["static_equity_best"]]} | {entry["pick"]}
    return {
        **view,
        "generation": generation,
        "candidates": analysis["candidates"],
        "pick": {**move_json(entry["pick"], board), "reading": entry["reading"]},
        "num_values": len(entry["values"]),
        "values": values_json(entry["values"], named, board, truths or {}),
        "truth": truth_about,
        "match": None
        if match is None
        else {k: match[k] for k in ("score", "ci_half_width", "games", "opponent")},
    }


def position_payload(tag_key: tuple[str, str, str], generation: int, index: int) -> dict:
    tag, task, mount_root = tag_key
    paths = TagPaths(*tag_key)
    p = known_positions.load()[index]
    doc = truth.load()
    truths = truth.of_position(doc, p)
    leaf_params = tasks.read_params(workloads.get(task), tag, mount_root)
    return {
        **payload(
            board_view(p),
            generation,
            _analysis(tag_key, generation),
            _matches(paths).get(generation),
            truths,
            truth_json(doc, truths, leaf_params),
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


class HistoryHandler(_Base):
    """One move's value at one known position over the analyzed generations."""

    async def get(self):
        index = int(self.get_query_argument("position", "0"))
        positions = known_positions.load()
        if not 0 <= index < len(positions):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        args = (self.tag_key(), positions[index].id, self.get_query_argument("move"))
        series = await tornado.ioloop.IOLoop.current().run_in_executor(_EXECUTOR, history, *args)
        self.write({"history": series})


ROUTES = [
    (r"/api/deep_bot_leaf_eval/generations", GenerationsHandler),
    (r"/api/deep_bot_leaf_eval/position", PositionHandler),
    (r"/api/deep_bot_leaf_eval/history", HistoryHandler),
]
