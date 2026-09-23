"""Incremental (append-only) updates for the dashboard's embedded figures.

The client embeds a figure from a full ``/api/figure`` fetch. When the run
advances, re-embedding the whole document would cost up to a megabyte on a long
run. Instead the client posts each named data source's length and last x. The
server rebuilds the figure model from SQLite (the cheap part) and answers with
each source's new tail rows plus the current explicit axis ranges, which the
client streams in place (web/src/lib/bokehDoc.ts).

Appends are exact because every plotted column is causal in x: raw series, the
debiased EMA, and the stacked-band cumulative sums (normalized or not) leave
earlier points unchanged when rows are appended. Any other change (a new
series, a control marker, recorded loss weights, a reset rewriting history)
shows up as a structure-key or tail mismatch, and the response is
``{"refetch": true}``: the client falls back to a full fetch.
"""

import hashlib

import numpy as np
from bokeh.models import ColumnDataSource, Plot, Range1d, Span


def _named_sources(model) -> dict:
    """name -> the model's ColumnDataSources with that name. A name can have
    several: the linear and log x-axis rows share names and hold identical
    data."""
    out: dict[str, list] = {}
    for cds in model.select({"type": ColumnDataSource}):
        if cds.name:
            out.setdefault(cds.name, []).append(cds)
    return out


def _row_panels(child) -> list:
    """A figure row's Plot panels in layout order, which is how the client zips
    them. Bokeh's `select` would return them in no fixed order."""
    return [p for p in getattr(child, "children", []) if isinstance(p, Plot)]


def structure_key(model) -> str:
    """A fingerprint of everything about the figure that appends cannot change:
    the named-source set, each row's panel titles, and the control-marker count.
    The client echoes it back; a mismatch means the document must be rebuilt."""
    names = sorted(_named_sources(model))
    rows = [
        (child.name, [p.title.text for p in _row_panels(child)])
        for child in getattr(model, "children", [])
    ]
    markers = len(list(model.select({"type": Span})))
    return hashlib.md5(repr((names, rows, markers)).encode()).hexdigest()


def _tail(data: dict, n: int, last_x) -> dict | None:
    """The rows of `data` past the client's first `n`, or None when the client's
    view cannot be extended in place (it is longer than ours, or its last x does
    not match ours at that index -- history was rewritten)."""
    x = np.asarray(data["x"], dtype=np.float64)
    if n > len(x) or n < 0 or (n == 0) != (last_x is None):
        return None
    if n and not np.isclose(x[n - 1], float(last_x), equal_nan=True):
        return None
    return {col: [float(v) for v in np.asarray(vals)[n:]] for col, vals in data.items()}


def _explicit_ranges(model) -> dict:
    """Per figure row (by name), each panel's explicit Range1d endpoints in layout
    order: {"x": [start, end] | None, "y": ...}. Auto ranges are None -- BokehJS
    recomputes those from the streamed data itself; explicit ones (the log rows'
    x, the padded y of learning curves) must be moved by the client."""

    def endpoints(rng):
        return [rng.start, rng.end] if isinstance(rng, Range1d) else None

    return {
        child.name: [
            {"x": endpoints(p.x_range), "y": endpoints(p.y_range)} for p in _row_panels(child)
        ]
        for child in getattr(model, "children", [])
    }


def delta_response(model, client: dict) -> dict:
    """The appends that bring the client's document up to `model`, or
    {"refetch": true} when appends alone cannot. `client` carries the
    document's structure key and, per source, {"n", "last_x"}."""
    if model is None or client.get("structure") != structure_key(model):
        return {"refetch": True}
    sources = _named_sources(model)
    states = client.get("sources") or {}
    if set(states) != set(sources):
        return {"refetch": True}
    tails = {}
    for name, copies in sources.items():
        state = states[name]
        tail = _tail(copies[0].data, int(state["n"]), state.get("last_x"))
        if tail is None:
            return {"refetch": True}
        tails[name] = tail
    return {"sources": tails, "ranges": _explicit_ranges(model)}
