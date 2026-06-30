# React dashboard + lane-analysis tool

This document specifies the migration of the training dashboard from a Bokeh-server
app to a **React app with a Python data API**, and the **max-move-per-lane lane
analysis** tab that this migration enables. The lane-analysis tab needs a genuinely
interactive board (click a square, highlight its lane), which a Bokeh-served page
cannot provide; a React shell that *embeds* the existing Bokeh plots removes that
limitation without rewriting the plots.

## Why React + embedded Bokeh

The interactive Scrabble board already exists as a React component (`web/src/components/Board.tsx`),
served by the C++ tools (`play_game`, `manual_gcg_tool`, `board_tool`) over a
WebSocket. The metrics dashboard, by contrast, is a Bokeh-server app. Those two
stacks cannot share a page, so an interactive board cannot live in the Bokeh
dashboard.

Bokeh 3.9 supports **standalone embedding**: `bokeh.embed.json_item(model)`
serializes any figure/layout to a JSON dict, and BokehJS renders it client-side
with `Bokeh.embed.embed_item(item, divId)`. A small React `<BokehFigure item={…}/>`
(mount a div, call `embed_item` in an effect) hosts the *existing* Bokeh plots
inside React. The interactive plots (position scrubber, generation slider, stacked
loss) are already built with **CustomJS** callbacks, which serialize into the
standalone embed and keep working. Only the few *server-side* `on_change` callbacks
(tag select, %/abs toggle, follow-latest, tab switch) move into React state.

BokehJS in `web/` is pinned to **3.9.1** to match the Python `bokeh` that produces
the `json_item`s.

## Architecture

```
              ┌───────────────────────── React app (Vite, VITE_TOOL=dashboard) ─────────────┐
  browser ──▶ │  shell: tag select · tab nav · 3s poll                                       │
              │   ├─ metrics tabs:  <BokehFigure item={…}/>   (embedded json_item)           │
              │   └─ lane-analysis tab: Board.tsx (native, interactive)                       │
              └───────────────▲───────────────────────────────────────────────▲──────────────┘
                              │  GET /api/...  (proxied through Vite)           │
              ┌───────────────┴──────────── Python data API (Tornado) ─────────┴──────────────┐
              │  reads dashboard.db (SQLite) · builds Bokeh json_items (plots.py)              │
              │  lane analysis: engine FFI (ground truth) + torch checkpoint eval (predictions)│
              └───────────────────────────────────────────────────────────────────────────────┘
```

- **Tornado** hosts the API. It is already present (a Bokeh dependency), so no new
  Python dependency is needed; FastAPI/Flask are not installed.
- The API is task-aware via query params (`task`, `tag`); one server serves any
  run. It opens a `dashboard.db` connection per request via
  `TagPaths(tag, task, mount_root)`.
- In dev, a launcher starts Tornado + the Vite dev server (`VITE_TOOL=dashboard`,
  mirroring how the C++ tools launch Vite); `vite.config.ts` proxies `/api` to
  Tornado (as it already proxies `/ws`), so the browser sees one origin.
- The C++ web tools are unchanged. The dashboard is a separate React app with a
  Python backend; it only *shares* the `web/src/components` library.

### API endpoints (Phase A)

| Endpoint | Returns |
|---|---|
| `GET /api/tags?task=<t>` | `{"tags": [...]}` — `db.list_tags(mount_root, task)` |
| `GET /api/version?task=<t>&tag=<g>` | `{"<table>": <row_count>, ...}` — cheap change token for polling (mirrors the Bokeh shell's `watch()`) |
| `GET /api/figure/<name>?task=<t>&tag=<g>&<params>` | a Bokeh `json_item` dict, or `{"item": null}` when there's no data |

`figure/<name>` dispatches to the existing builders in `plots.py` (e.g.
`train_step` → `train_step_grid(conn, normalized)`), serialized with `json_item`.
The builders are reused unchanged.

## Status

The migration is **complete** — the dashboard is the React app; there is no longer
a Bokeh-served dashboard. It was done in three phases:

- **Phase A — foundation (done).** Python API (tags, version, `train_step` figure) +
  React shell (tag select, polling, `<BokehFigure>`).
- **Phase B — lane analysis (done).** The interactive board tab (below).
- **Phase C — migrate the rest (done).** The post-move tabs (Loss, Positions,
  Calibration, Training, Performance) are React tabs that embed `json_item`s built
  by the `plots.py` builders; both post-move trainers and the standalone launcher
  (`scripts/dashboard.py`) launch the React dashboard. The Bokeh-serving modules
  (`server.py`, `shell.py`, `post_move_tabs.py`, the `app_*.py`) were deleted;
  `plots.py` and `db.py` remain (the API reuses them).

### Tabs (task-conditional, all in `web/src/AppDashboard.tsx`)

| Task | Tabs |
|---|---|
| post_move_value | Loss · Positions · Calibration · Training · Performance |
| max_move_per_lane | Loss · Performance · Lane analysis |

Each non-`Lane analysis` tab is a `<FigureTab>` that embeds an API figure
(`train_step`, `throughput`, `training_metrics`, `positions`, `calibration`) and
re-fetches when its version-token table advances. `Lane analysis` is native React.

## Phase B — lane-analysis tab

A native-React tab that visualizes how each model generation evaluates each
position in a GCG dataset (default `positions/NWL23/max-move-per-lane-test-dataset/`).

### The analysis position (per `pos-N.gcg`)

Each file is a full game (the `>` move lines) that builds a realistic board. The
`#Rack1`/`#Rack2` headers are the racks at the **final** position (after all
recorded moves), where the on-move player — determined by move-count parity — holds
the full 7-tile rack and the opponent's is partial. So each file defines exactly
**one** analysis position: the final board, the next player to move, fed their full
`#RackX` rack.

### Ground truth (C++ engine)

Per position: replay the GCG to the final board; take the on-move rack; then
`compute_lane_targets` (per-lane union / max score / has-move) **plus a new per-lane
best-move enumeration** — re-run `MoveGenerator`, keep the moves tied for the
lane max, and return their word / coordinates / direction / score.
`compute_lane_targets` alone discards the moves (it keeps only the union), so the
enumeration is new.

### Model predictions (Python, in-process at each checkpoint)

- One new FFI: encode a GCG final-position (board + rack) into the model input
  tensor (`MaxMovePerLaneInputEncoder`). A one-time builder caches the dataset's
  input-tensor batch under the tag.
- The trainer's checkpoint hook runs the in-memory model on that batch, decodes per
  lane (predicted occupancy union, 100-bin score PMF, has-move), and writes
  per-generation predictions (occupancy union, score PMF, has-move) into the tag's
  `dashboard.db` (`lane_pred` table), keyed by generation. Weights are not retained
  (a generation = one checkpoint's predictions).

### UI (reuses `Board.tsx`)

- Generation slider + "latest" checkbox; position dropdown + ◀/▶; H/V radio.
- `Board.tsx` extensions: per-row/column ✓/✗ badges (green = move subtask, blue =
  score subtask, for lanes with a legal move); a `selectedLane` highlight; click a
  square → highlight that square and its lane (orientation from the radio).
- Lane-detail pane: the true best move(s) (word / coords / score) and the
  true-vs-predicted tile-union diff, colored per cell — **green** = correct tile,
  **amber** = a true tile the model missed, **red** = a tile the model
  hallucinated.
- Score histogram: the selected lane's predicted 100-bin score PMF with the true
  score bin highlighted.
