"""Data for the Positions tab of deep_bot_leaf_eval and of
leaf_bot_generation: one known position at a time under one generation's
analysis (deep_bot_leaf_eval/positions.py), which a match slot delivered
with that generation's match against HastyBot. A leaf_bot_generation tag's
"generation" is a release. Registered with the rest of the data plane by
api.make_app().

Routes (all but the first take task + tag):
  /api/known_positions                 the known positions, in manifest order
  /api/deep_bot_leaf_eval/generations  the analyzed generations (the slider),
                                       with their rows trained and match score
  /api/deep_bot_leaf_eval/position     ?position=&generation= -> the board,
                                       the matches' pick there, and the leaf's
                                       value of the moves worth listing, beside
                                       their hasty rollouts' (deep_bot_leaf_eval/
                                       hasty_rollouts.py) where the repo holds
                                       them
  /api/deep_bot_leaf_eval/history      ?position=&move= -> the leaf's value of
                                       one move under every analyzed generation
  /api/deep_bot_leaf_eval/belief       ?position= -> the mover's belief about
                                       the opponent's rack under the tag's
                                       fixed leave inference model
                                       (deep_bot_leaf_eval/opp_beliefs.py),
                                       charted on the first request and kept

An analysis is read again only when its file changes, which happens when the
positions backfill (workloads/positions_backfill.py) adds the positions the
manifest gained: the slider's from its file, a history's from the tag's
leaf_value table (dashboard/db.py), which a history request first brings up
to date with the analyses delivered or rewritten since.
"""

import functools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tornado.ioloop
import tornado.web

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import db, tasks
from scribblez.deep_bot import known_positions
from scribblez.deep_bot.matches import HASTYBOT
from scribblez.deep_bot.position_view import board_view, move_json
from scribblez.deep_bot_leaf_eval import hasty_rollouts, opp_beliefs
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.paths import TagPaths
from scribblez.sim_survey_viewer import TILE_SCORES, bonuses
from scribblez.workloads import deep_bot_leaf_eval, leaf_bot_generation

# The likeliest moves under the leaf the table lists, besides any that the
# reference, static equity or the pick names.
SHOWN_VALUES = 25
# The hasty rollouts' best moves the table lists too.
SHOWN_ROLLOUTS = 10
# A history request first indexes the analyses changed since the last, every
# one on a tag's first request (~8 s for 127 generations), so it runs off the
# IOLoop; one thread, so no two requests index the same generation.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)
# A belief request first charts the positions not kept yet (opp_beliefs.charts),
# so it runs off the IOLoop too, behind no history indexing; one thread, so no
# two requests chart the same positions.
_BELIEF_EXECUTOR = ThreadPoolExecutor(max_workers=1)
# Each workload's leave inference model the opponent's rack is charted under,
# with its label (its belief_model).
_BELIEF_MODELS = {
    deep_bot_leaf_eval.SPEC.name: deep_bot_leaf_eval.belief_model,
    leaf_bot_generation.SPEC.name: leaf_bot_generation.belief_model,
}


def _stamp(path: Path) -> tuple[int, int]:
    """A file's (mtime_ns, size), which a rewrite changes."""
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _analysis(tag_key: tuple[str, str, str], generation: int) -> dict:
    path = TagPaths(*tag_key).data_dir / leaf_positions.rel(generation)
    return _read_analysis(path, _stamp(path))


# Keyed by the file's stamp too, so a rewritten analysis is read afresh. A
# missing file raises, which is not cached.
@functools.lru_cache(maxsize=32)
def _read_analysis(path: Path, stamp: tuple[int, int]) -> dict:
    return json.loads(path.read_text())


def _values(path: Path) -> list[tuple[str, str, float]]:
    """Every (position, move, value) of the analysis at `path`."""
    doc = json.loads(path.read_text())
    return [
        (position, v["move"], v["value"])
        for position, entry in doc["positions"].items()
        for v in entry["values"]
    ]


def sync_values(paths: TagPaths, conn):
    """Index into leaf_value every analysis it does not hold as its file now
    is. The stamp is taken before the read, so a rewrite between them is
    indexed on the next sync."""
    held = db.read_leaf_value_stamps(conn)
    for g in leaf_positions.analyzed(paths):
        path = paths.data_dir / leaf_positions.rel(g)
        stamp = _stamp(path)
        if held.get(g) != stamp:
            db.write_leaf_values(conn, g, stamp, _values(path))


def history(tag_key: tuple[str, str, str], position: str, move: str) -> list[dict]:
    """The leaf's value of `move` at `position` under each analyzed
    generation, ascending; null under one that did not value it."""
    paths = TagPaths(*tag_key)
    conn = db.connect(paths.dashboard_db)
    try:
        sync_values(paths, conn)
        return db.read_leaf_value_history(conn, position, move)
    finally:
        conn.close()


def belief(tag_key: tuple[str, str, str], position: str) -> dict:
    """The opponent-rack chart at `position` under the tag's fixed leave
    inference model (web RackBeliefChart)."""
    tag, task, mount_root = tag_key
    spec = workloads.get(task)
    params = params_mod.validate(spec.params_cls, tasks.read_params(spec, tag, mount_root))
    paths = TagPaths(*tag_key)
    leave, label = _BELIEF_MODELS[task](paths, params)
    return {**opp_beliefs.charts(paths, leave)[position], "model": label}


def _matches(paths: TagPaths) -> dict[int, dict]:
    """The tag's match results against HastyBot by generation, from
    deep_bot_leaf_eval's match eval or by release from leaf_bot_generation's
    release matches; none before its first."""
    if not paths.dashboard_db.is_file():
        return {}
    conn = db.connect(paths.dashboard_db)
    try:
        by_release = {
            r["release"]: r for r in db.read_release_matches(conn) if r["opponent"] == HASTYBOT
        }
        return by_release | {r["epoch"]: r for r in db.read_all_match_eval(conn)}
    finally:
        conn.close()


def generation_list(paths: TagPaths) -> list[dict]:
    """The slider's entries: each analyzed generation, its rows trained (from
    its match row) and its match score, when the match is recorded."""
    matches = _matches(paths)
    out = []
    for g in leaf_positions.analyzed(paths):
        m = matches.get(g)
        out.append(
            {
                "generation": g,
                "rows_trained": m.get("positions") if m else None,
                "score": m["score"] if m else None,
                "ci_half_width": m["ci_half_width"] if m else None,
            }
        )
    return out


def rollout_ranks(values: list[dict], rollouts: dict[str, float]) -> dict[str, int]:
    """Each move's rank under the hasty `rollouts`, the better static equity
    first on ties, over the leaf's `values` the rollouts cover."""
    covered = sorted(
        (v for v in values if v["move"] in rollouts),
        key=lambda v: (-rollouts[v["move"]], v["equity_rank"]),
    )
    return {v["move"]: i + 1 for i, v in enumerate(covered)}


def values_json(
    values: list[dict], named: set[str], board, rollouts: dict[str, float]
) -> list[dict]:
    """The value table's rows, best first: the SHOWN_VALUES best under the
    leaf, the SHOWN_ROLLOUTS best under the hasty rollouts and any move in
    `named`, each drawn on the pre-move `board` and with its rollout value
    (null where it has none)."""
    ranks = rollout_ranks(values, rollouts)
    return [
        {
            **v,
            **move_json(v["move"], board),
            "rollout": rollouts.get(v["move"]),
            "rollout_rank": ranks.get(v["move"]),
        }
        for v in values
        if v["leaf_rank"] <= SHOWN_VALUES
        or ranks.get(v["move"], SHOWN_ROLLOUTS + 1) <= SHOWN_ROLLOUTS
        or v["move"] in named
    ]


def rollouts_json(doc: dict | None, rollouts: dict | None, leaf_params: dict | None) -> dict | None:
    """What the table says about the position's hasty rollouts, from the
    repo's file `doc`: whose they are and how they were run, whether they drew
    the opponent's leaves from the tag's analysis's belief, and how many moves
    they cover; None without them."""
    if rollouts is None:
        return None
    return {
        "tag": doc["tag"],
        "run": doc["run"],
        "same_belief": hasty_rollouts.same_belief(doc["run"], leaf_params or {}),
        "num_values": len(rollouts),
    }


def payload(
    view: dict,
    generation: int,
    analysis: dict,
    match: dict | None,
    rollouts: dict[str, float] | None = None,
    rollouts_about: dict | None = None,
) -> dict:
    """The tab's view of a known position under one generation: its
    board_view, the matches' pick there and how it reads against the answer,
    and the leaf's values of the moves worth listing beside their hasty
    `rollouts` (hasty_rollouts.of_position), which `rollouts_about`
    (rollouts_json) describes."""
    entry = analysis["positions"][view["id"]]
    board = view["board"]
    named = {m["move"] for m in [*view["reference"], view["static_equity_best"]]} | {entry["pick"]}
    return {
        **view,
        "generation": generation,
        "candidates": analysis["candidates"],
        "pick": {**move_json(entry["pick"], board), "reading": entry["reading"]},
        "num_values": len(entry["values"]),
        "values": values_json(entry["values"], named, board, rollouts or {}),
        "rollouts": rollouts_about,
        "match": None
        if match is None
        else {k: match[k] for k in ("score", "ci_half_width", "games", "opponent")},
    }


def position_payload(tag_key: tuple[str, str, str], generation: int, index: int) -> dict:
    tag, task, mount_root = tag_key
    paths = TagPaths(*tag_key)
    p = known_positions.load()[index]
    doc = hasty_rollouts.load()
    rollouts = hasty_rollouts.of_position(doc, p)
    leaf_params = tasks.read_params(workloads.get(task), tag, mount_root)
    return {
        **payload(
            board_view(p),
            generation,
            _analysis(tag_key, generation),
            _matches(paths).get(generation),
            rollouts,
            rollouts_json(doc, rollouts, leaf_params),
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


class KnownPositionsHandler(tornado.web.RequestHandler):
    def get(self):
        positions = known_positions.load()
        self.write({"positions": [{"id": p.id, "title": p.title} for p in positions]})


class GenerationsHandler(_Base):
    def get(self):
        self.write({"generations": generation_list(TagPaths(*self.tag_key()))})


class PositionHandler(_Base):
    """One known position under one analyzed generation; `generation`
    defaults to the newest."""

    def get(self):
        analyzed = leaf_positions.analyzed(TagPaths(*self.tag_key()))
        arg = self.get_query_argument("generation", "latest")
        generation = (analyzed[-1] if analyzed else None) if arg in ("", "latest") else int(arg)
        if generation not in analyzed:
            self.set_status(404)
            self.write({"error": "no such analysis" if analyzed else "no analysis yet"})
            return
        index = int(self.get_query_argument("position", "0"))
        positions = known_positions.load()
        if not 0 <= index < len(positions):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        # A position added to the manifest after the generation was analyzed.
        if positions[index].id not in _analysis(self.tag_key(), generation)["positions"]:
            self.set_status(404)
            self.write({"error": f"generation {generation} was analyzed before this position"})
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


class BeliefHandler(_Base):
    """The opponent-rack belief at one known position."""

    async def get(self):
        index = int(self.get_query_argument("position", "0"))
        positions = known_positions.load()
        if not 0 <= index < len(positions):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        args = (self.tag_key(), positions[index].id)
        chart = await tornado.ioloop.IOLoop.current().run_in_executor(
            _BELIEF_EXECUTOR, belief, *args
        )
        self.write({"belief": chart})


ROUTES = [
    (r"/api/known_positions", KnownPositionsHandler),
    (r"/api/deep_bot_leaf_eval/generations", GenerationsHandler),
    (r"/api/deep_bot_leaf_eval/position", PositionHandler),
    (r"/api/deep_bot_leaf_eval/history", HistoryHandler),
    (r"/api/deep_bot_leaf_eval/belief", BeliefHandler),
]
