"""Tests for the figures' incremental-update path (figure_delta.py): appends are
exact, and every structural or historical change falls back to a full refetch."""

import numpy as np
from bokeh.models import ColumnDataSource
from scribblez.dashboard import db, figure_delta, plots


def _write_epoch(conn, epoch, pos, loss=1.0):
    db.write_metrics(
        conn, epoch, {"positions": pos, "loss": loss, "eval_win_mae": 0.1 / (epoch + 1)}
    )


def _client_state(model):
    """What the web client would post: the structure key plus each named source's
    length and last x, read from its embedded copy of `model`."""
    sources = {}
    for cds in model.select({"type": ColumnDataSource}):
        if cds.name:
            x = list(cds.data["x"])
            sources[cds.name] = {"n": len(x), "last_x": float(x[-1]) if x else None}
    return {"structure": figure_delta.structure_key(model), "sources": sources}


def _grown(tmp_path, build, n_before=3, n_after=5):
    """(client state at n_before epochs, model at n_after epochs, conn)."""
    conn = db.connect(tmp_path / "d.db")
    for e in range(n_before):
        _write_epoch(conn, e, 100 * (e + 1))
    state = _client_state(build(conn))
    for e in range(n_before, n_after):
        _write_epoch(conn, e, 100 * (e + 1))
    return state, build(conn), conn


def test_append_matches_full_rebuild(tmp_path):
    state, model, _ = _grown(tmp_path, plots.metrics_loss_grid)
    resp = figure_delta.delta_response(model, state)
    assert "refetch" not in resp
    assert set(resp["sources"]) == set(state["sources"])
    for name, tail in resp["sources"].items():
        # Appending the tail to the client's copy reproduces the rebuilt source.
        rebuilt = next(iter(model.select({"type": ColumnDataSource, "name": name}))).data
        n = state["sources"][name]["n"]
        for col, vals in tail.items():
            assert len(vals) == len(rebuilt[col]) - n
            assert np.allclose(vals, np.asarray(rebuilt[col])[n:])


def test_append_is_exact_under_smoothing(tmp_path):
    """The EMA is causal: streamed tail values equal the full rebuild's."""

    def build(conn):
        return plots.eval_quality_grid(conn, "t", smooth=True)

    conn = db.connect(tmp_path / "d.db")
    rng = np.random.default_rng(0)
    for e in range(plots._SMOOTH_MIN_POINTS + 2):
        db.write_metrics(conn, e, {"positions": 100 * e, "eval_win_mae": float(rng.random())})
    state = _client_state(build(conn))
    for e in range(plots._SMOOTH_MIN_POINTS + 2, plots._SMOOTH_MIN_POINTS + 6):
        db.write_metrics(conn, e, {"positions": 100 * e, "eval_win_mae": float(rng.random())})
    model = build(conn)
    resp = figure_delta.delta_response(model, state)
    assert "refetch" not in resp
    for name, tail in resp["sources"].items():
        rebuilt = next(iter(model.select({"type": ColumnDataSource, "name": name}))).data
        assert np.allclose(tail["y"], np.asarray(rebuilt["y"])[state["sources"][name]["n"] :])


def test_ranges_cover_explicit_axes_only(tmp_path):
    state, model, _ = _grown(tmp_path, plots.metrics_loss_grid)
    ranges = figure_delta.delta_response(model, state)["ranges"]
    assert set(ranges) == {
        f"{x}|{n}"
        for x in (plots.X_AXIS_LINEAR, plots.X_AXIS_LOG)
        for n in (plots.NORM_ABSOLUTE, plots.NORM_PERCENT)
    }
    for name, panels in ranges.items():
        for panel in panels:
            if name.startswith(plots.X_AXIS_LOG):
                assert panel["x"] is not None and 0.0 < panel["x"][0] < panel["x"][1]
            else:
                assert panel["x"] is None  # auto -- BokehJS follows the streamed data


def test_new_series_forces_refetch(tmp_path):
    state, _, conn = _grown(tmp_path, plots.metrics_loss_grid)
    db.write_metrics(conn, 5, {"positions": 600, "loss": 1.0, "loss_new": 0.5})
    assert figure_delta.delta_response(plots.metrics_loss_grid(conn), state) == {"refetch": True}


def test_control_marker_forces_refetch(tmp_path):
    state, _, conn = _grown(tmp_path, plots.metrics_loss_grid)
    db.write_control_event(conn, 450, "lr", 1e-4)
    assert figure_delta.delta_response(plots.metrics_loss_grid(conn), state) == {"refetch": True}


def test_run_reset_forces_refetch(tmp_path):
    """A recreated DB (shorter history) cannot be reached by appends."""
    state, _, _ = _grown(tmp_path, plots.metrics_loss_grid, n_before=3, n_after=3)
    conn2 = db.connect(tmp_path / "d2.db")
    _write_epoch(conn2, 0, 100)
    assert figure_delta.delta_response(plots.metrics_loss_grid(conn2), state) == {"refetch": True}


def test_rewritten_history_forces_refetch(tmp_path):
    """Same lengths and structure, different x at the client's cursor."""
    state, model, _ = _grown(tmp_path, plots.metrics_loss_grid, n_before=3, n_after=3)
    state["sources"] = {
        name: {**s, "last_x": (s["last_x"] or 0.0) + 7.0} for name, s in state["sources"].items()
    }
    assert figure_delta.delta_response(model, state) == {"refetch": True}


def test_no_growth_is_an_empty_append(tmp_path):
    state, model, _ = _grown(tmp_path, plots.metrics_loss_grid, n_before=3, n_after=3)
    resp = figure_delta.delta_response(model, state)
    assert "refetch" not in resp
    assert all(all(len(v) == 0 for v in tail.values()) for tail in resp["sources"].values())


def test_missing_model_forces_refetch():
    assert figure_delta.delta_response(None, {"structure": "x", "sources": {}}) == {"refetch": True}
