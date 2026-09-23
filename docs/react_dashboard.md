# Training-analysis tabs

The master dashboard ([master_dashboard.md](master_dashboard.md)) is a React
app over a Python data API. For the training workloads it adds analysis tabs:
Bokeh metric figures embedded in React, plus three native interactive boards
(**Positions**, **Trajectories**, **Lane analysis**) described below.

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
| max_move_per_lane | Loss · Lane analysis · Controls · Info |
| move_set_eval | Loss · Training · Controls · Info |
| evidence_trajectories | Loss · Trajectories · Match · Training · Controls · Info |
| match_arms | Arms · Info |

Loss, Match, Training and Arms embed API figures and re-fetch when a cheap
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

## Trajectories (evidence_trajectories)

Shows the sequential evidence loop working on a hand-maintained position set
([positions/NWL23/face-up-trajectory-set](../positions/NWL23/face-up-trajectory-set/README.md)),
and how the trained model responds to it.

- **Position sets.** A `.gcg` is the position: its final recorded state, the
  side to move next, and that side's rack from a `#RackN` pragma. The
  trajectory sidecars are simmed on first request under the tag's own proposer
  and recipe (`sim_evidence.position_sets.ensure_sobs`, cached under
  `<mount>/cache/trajectory_sets/`), so the tab shows exactly the trajectory
  the generator would produce there.
- **Model.** Generation 0 is the frozen student itself (the tag's
  `student_checkpoint`). Generation N is the trainer's torch checkpoint
  `checkpoints/model_epoch_{N-1}.pt`, run in torch on CPU
  (`scribblez.evidence.trajectory_view`). The FFI (`gcg_position_inputs`)
  rebuilds the decision as the generator saw it: the board row under the
  checkpoint's input arm, the pre-move score differential, and the full
  equity-ranked legal move list. The plain (evidence-free) pass runs once per
  (checkpoint, position); only the fusion stage and re-score run per evidence
  prefix.
- **UI.**
  - Generation slider, set and position pickers, and an **evidence prefix**
    slider from 0 to the number of on-policy picks (off-policy draws are never
    evidence).
  - The trajectory strip (anchor → on-policy → off-policy, dimmed beyond the
    prefix). Each card shows its sim value ± SE, the delta moments, and the
    plain and conditioned values.
  - The board, previewing the selected candidate with the Positions tab's
    placement overlay: sim count planes vs. the conditioned pass's planes, as
    residual, model, or sim.
  - The legal moves re-ranked by conditioned value, with the shift from the
    plain rank, the proves-best gain, and the loop's **next sim** (the
    unsimmed move with the highest gain) marked.

  At prefix 0 the conditioned pass equals the plain one by construction; at
  generation 0 conditioning is the identity and there is no gain column.
- The trainer's `posset_*` metrics (Training tab) score the same set at every
  prefix: the rank of the sim-best candidate under conditioned vs. plain
  value. What that chart summarizes is what this tab shows.

![The Trajectories tab on the egotize-lane exhibit: the anchor's opp-play-and-win residual lights the 13G–13N lane red (the model under-reads the opponent's EGOTIZE lane), the trajectory strip, and the conditioned re-ranking](images/trajectories_tab.png)

## Lane analysis (max_move_per_lane)

Shows how each generation solves the max-move-per-lane task
([lexical_nn.md](lexical_nn.md)) over a GCG dataset
(`positions/NWL23/max-move-per-lane-test-dataset/`).

- **Dataset contract.** Each `pos-N.gcg` is a full game whose
  `#Rack1`/`#Rack2` headers are the racks at the final position. The player
  on move (by move-count parity) holds the full 7-tile rack, which defines
  exactly one analysis position per file.
- **Ground truth.** Replay to the final board, then compute per-lane targets
  and enumerate each lane's best moves (all moves tied for the lane's max,
  with word, coordinates and score).
- **Predictions.** The trainer builds the dataset's input batch once through
  the FFI and caches it under the tag. After each checkpoint it decodes
  per-lane predictions (occupancy union, score PMF, has-move) into the
  generation's record, which the dashboard ingests into `dashboard.db`.
- **UI.** Generation slider, position picker, and orientation radio;
  `Board.tsx` with per-lane pass/fail badges and lane highlighting; a lane
  detail pane with the true best move(s) and the true-vs-predicted tile-union
  diff; and the selected lane's predicted score PMF with the true bin
  highlighted.
