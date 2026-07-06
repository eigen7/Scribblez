# React dashboard + lane-analysis tool

The training dashboard is a **React app with a Python data API**: a React shell
(tabs, tag select, polling) that embeds the Bokeh metric figures and hosts the
genuinely interactive tabs — the **max-move-per-lane lane analysis** board
(click a square, highlight its lane) and the **post-move-value Positions**
board — as native React components.

## Why React + embedded Bokeh

The interactive Scrabble board is a React component (`web/src/components/Board.tsx`),
served by the C++ tools (`play_game`, `manual_gcg_tool`, `board_tool`) over a
WebSocket. A Bokeh-served page cannot host it, so the page structure is React,
and the metric plots are embedded into it.

Bokeh 3.9 supports **standalone embedding**: `bokeh.embed.json_item(model)`
serializes any figure/layout to a JSON dict, and BokehJS renders it client-side
with `Bokeh.embed.embed_item(item, divId)`. A small React `<BokehFigure item={…}/>`
(mount a div, call `embed_item` in an effect) hosts the Bokeh plots inside React.
The interactive plots (position scrubber, generation slider, stacked loss) are
built with **CustomJS** callbacks, which serialize into the standalone embed and
work client-side; page-level state (tag select, %/abs toggle, follow-latest, tab
switch) lives in React.

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

### API endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/tags?task=<t>` | `{"tags": [...]}` — `db.list_tags(mount_root, task)` |
| `GET /api/version?task=<t>&tag=<g>` | `{"<table>": <row_count>, ...}` — cheap change token for polling |
| `GET /api/figure/<name>?task=<t>&tag=<g>&<params>` | a Bokeh `json_item` dict, or `{"item": null}` when there's no data |

`figure/<name>` dispatches to the builders in `plots.py` (e.g.
`loss` → `metrics_loss_grid(conn, normalized)`), serialized with `json_item`.
Both trainers and the standalone launcher (`scripts/dashboard.py`) launch the
React dashboard.

### Tabs (task-conditional, all in `web/src/AppDashboard.tsx`)

| Task | Tabs |
|---|---|
| post_move_value | Loss · Positions · Training · Controls · Info |
| max_move_per_lane | Loss · Lane analysis · Info |

`Loss` and `Training` embed an API figure (`loss`, `training_metrics`) and
re-fetch when their version-token table advances. `Positions` (post-move-value)
and `Lane analysis` (max-move-per-lane) are native-React interactive tabs (see
below).

## The lane-analysis tab

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
`compute_lane_targets` (per-lane union / max score / has-move) **plus a per-lane
best-move enumeration** — re-run `MoveGenerator`, keep the moves tied for the
lane max, and return their word / coordinates / direction / score
(`compute_lane_targets` alone keeps only the union, not the moves).

### Model predictions (Python, in-process at each checkpoint)

- One FFI: encode a GCG final-position (board + rack) into the model input
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

## The post-move-value Positions tab

A native-React tab that compares each model generation's post-move-value
prediction against a Monte-Carlo ground truth, over a GCG dataset
(`positions/NWL23/post-move-value-test-dataset/`).

### The analysis position (per `pos-N.gcg`)

Each file's penultimate move is a bingo, so the player to act next drew a clean
full rack. The analysis position is the board **after the final recorded move**,
evaluated from the POV of the player that made it (the "start player"), whose
**leave** (final `rack_before` minus the placed tiles) is its rack.

### Ground truth (offline Monte-Carlo, committed)

`engine/apps/monte_carlo_sim_tool.cpp` plays each position out N≈10k times —
HastyBot vs HastyBot, `Game::play_from` seeding the unknown racks/bag per game
`g` (deterministic, thread-independent) — and records the exact W/L/D and
final-score-delta histogram from the start player's POV into
`monte-carlo-sim-results.json`, committed beside the GCGs.

### Model predictions (Python, in-process at each checkpoint)

- One FFI (`scribblez_post_move_value_analyze_gcg`): replay the GCG into a fresh
  `GameStateEncoder` and `encode_input` from the start player's POV — byte-identical
  to a training row's input. A board-bundle FFI
  (`scribblez_post_move_value_board_json`) serves the renderable board (leave rack).
- The trainer's checkpoint hook runs the model on the dataset batch and writes each
  generation's WLD probabilities + score-delta mean/std into the tag's
  `dashboard.db` (`post_move_pred` table), keyed by generation.

### UI (reuses `Board.tsx`)

- Generation slider + "latest" checkbox; position dropdown + ◀/▶; the board + leave.
- WLD chart: model-vs-Monte-Carlo paired bars for win / loss / draw.
- Score-delta chart: the MC exact histogram (binned for display) with the model's
  predicted Gaussian (mean/std) overlaid as probability mass per bucket, plus both
  means marked.
