# Training-analysis tabs

The master dashboard ([master_dashboard.md](master_dashboard.md)) is a React
app over a Python data API. For the training workloads it adds analysis tabs:
Bokeh metric figures embedded in React, plus native interactive boards, of
which position_eval's **Positions** tab is described below.

## Why React with embedded Bokeh

The interactive Scrabble board is a React component
(`web/src/components/Board.tsx`), shared with the C++ web tools. A
Bokeh-served page cannot host it, so the page is React and the metric plots
are embedded through Bokeh's standalone embedding: `json_item` on the server,
`embed_item` in a small `<BokehFigure>` wrapper. Plot-internal interactivity
(scrubbers, sliders) is CustomJS, which survives standalone embedding;
page-level state (tag, toggles, tabs) lives in React.

The BokehJS version pinned in `web/package.json` must match the Python
`bokeh` that produces the `json_item`s. Upgrade them together.

## Architecture

Tornado (already installed as a Bokeh dependency) serves the data API. A
request names its workload and tag, and the server opens that tag's
`dashboard.db` (SQLite), so one server serves every run. `plots.py` builds the
Bokeh figures, and figure endpoints dispatch to its builders by name. In
development, Vite proxies `/api` to Tornado so the browser sees one origin.

Tabs are registered per workload in `web/src/workloads.tsx`; Overview and
Stats are generic and come from the task view itself.

| Workload | Tabs |
|---|---|
| position_eval | Loss · Positions · Match · Training · Controls · Info |
| match_arms | Arms · Info |
| leave_inference | Loss · Training · Controls · Info |
| deep_bot_leaf_eval | Loss · Training · Match · Positions · Controls · Info |
| deep_bot_generation | Loss · Training · Matches · Positions · Controls · Info |

Loss, Match, Matches, Training and Arms embed API figures and re-fetch when a cheap
version token advances.

## Positions (position_eval)

Compares each model generation's predictions against a committed Monte-Carlo
ground truth over a GCG dataset
(`positions/NWL23/position-eval-test-dataset/`).

- **Dataset contract** (see
  [the set's README](../positions/NWL23/position-eval-test-dataset/README.md)).
  Each `pos-N.gcg` is a post-move position: the board after the final recorded
  move, evaluated from the point of view of the player who made it, with their
  leave as the rack. The opponent, to act next, holds what their own last
  recorded move kept plus hidden draws. The engine reader is
  `read_gcg_post_move` in `engine/include/data/gcg_post_move.h`, shared by the
  ground-truth tool and the encoder.
- **Ground truth.** `monte_carlo_sim_tool` plays each position out ~10k times
  and commits W/L/D, the final-score-delta histogram, and the four placement
  planes beside the GCGs. It does so once per **information condition**,
  because the truth depends on what a rollout knows of the opponent's leave:
  - `monte-carlo-sim-results.face-up-leaves.json`: the leave is seated in
    every rollout.
  - `monte-carlo-sim-results.hidden-leaves.json`: the leave is inferred from
    the opponent's last move with `belief::RackInferrer` and sampled per
    rollout (uniform after a bingo).

  A tag is measured against the truth matching its `face_up_leaves` param.
- **Predictions** are computed on demand and never stored. An FFI replays the
  GCG into an input tensor byte-identical to a training row, and the selected
  generation's exported ONNX runs on it under fp32 onnxruntime. One forward
  pass yields the WLD and score-delta heads and the placement planes. The
  slider lists the generations that have an export. Nothing is keyed to the
  dataset's shape, so its files can be added, renamed or rewritten freely:
  the per-position caches key on each file's mtime, and a running dashboard
  follows the change.
- **UI.** Generation slider and position picker; the board with both racks
  (the POV player's leave, and the opponent's: spelled out under face-up
  leaves, "?" under hidden, plus their green-shaded hidden draws); a
  per-head placement overlay on the board showing the model's prediction, the
  MC truth, or their residual; paired model-vs-MC WLD bars; the MC score-delta
  histogram with the model's Gaussian overlaid; and an alternate-leaves
  what-if (model only) for either rack.
