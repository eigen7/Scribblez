# Collecting HastyBot's blind spots

A *blind spot* is a position where a play from outside HastyBot's top moves by
static equity out-sims all of them. The `blind_spots` dashboard workload collects
them on any number of machines at once; this page is how to run it and what
comes out. What the survey measures, and what it has found, is in
[plans/sim_labeled_candidates.md](plans/sim_labeled_candidates.md).

## Running it

1. In the dashboard, pick the workload **Collect HastyBot blind spots**, create
   a tag (the defaults are the settings the findings so far were made with),
   and add **Surveyor** workers: local, on an ssh machine, or on rented
   machines ([master_dashboard.md](master_dashboard.md), *Machines*). Ten
   rented CPU machines are ten workers. They need no coordination, since each
   plays its own randomly seeded games.
2. Start them. A worker's cycle plays one HastyBot-vs-HastyBot game and surveys
   every eligible turn of it, taking about ten minutes on 28 threads. The
   Overview shows positions found and games surveyed; Stats shows the rate per
   worker.
3. Walk away. When the tag holds `target_positions` found positions (a tag
   parameter, 100 by default; 0 runs until stopped), the controller parks
   every surveyor, shown as *waiting (target reached: …)*. Ten minutes later
   the idle policy **stops** the rented machines, which ends their compute
   charge. Stopped is not terminated: a stopped machine keeps its disk (cents a
   day) until you click Remove on its row. That is deliberate throughout the
   dashboard, which never deletes on its own a disk that might hold
   undelivered output; here such a disk holds only the game each worker was
   part-way through. The dashboard must be running for any of this, since it
   is the controller that counts, parks and stops.
4. Browse what came in, then turn it into a committed directory:

   ```
   ./py/scripts/sim_survey_viewer.py --tag <tag>
   ./py/scripts/blind_spots_collect.py --tag <tag> [--min-gain 8]
   git add positions/NWL23/best-bot-blind-spots
   ```

   `blind_spots_collect.py` rebuilds the directory from the tag each time, so
   rerun it as the tag grows. `--min-gain` drops plays whose edge over the
   best top move is under that many percentage points of win rate. The tag's
   own `min_gain` parameter (5 points by default) already applied that bar
   when the workers kept positions, so the flag only matters above it. Both
   exist because, with 5000-rollout confirming sims, the survey's
   two-standard-error bar alone admits edges well under one point.

## What a worker delivers

Into `/workspace/mount/tags/blind_spots/<tag>/data/`, by a rename for a local
worker, or collected over ssh from a remote one's container:

- `survey/<stem>.simsurvey.json`: one per game, slimmed to the positions
  found: those where an outside play beats the best top move by both the
  two-standard-error bar and `min_gain`. Each position lists the top moves
  and the outside picks with their confirming-sim summaries
  (`engine/include/sim/rollout_report.h`). The file also records
  `positions_surveyed`, the number of turns surveyed, so rates stay
  computable.
- `gcg/<stem>-g0-turn<N>.gcg`: the game up to each found position.

`<stem>` is `<timestamp>-<worker id>`, unique across workers. A game's full
survey (every legal play's screening summary, megabytes per game) never leaves
the worker and is discarded. To keep full surveys, run
`py/scripts/sim_candidate_survey.py` on one machine instead.

## Hidden-leaves blind spots on the GPU

The workload above sims on the CPU, with face-up leaves by default and the
win table or the endgame solver at its leaves. DeepBot's Alice evaluates
differently: hidden leaves, the opponent's leave drawn from the leave
inference model, two HastyBot replies, then the leaf network
([plans/deep_bot.md](plans/deep_bot.md#sims)). `blind_spot_confirm_tool`
finds blind spots under that evaluation, from pools
`policy_set_pool_tool` already labeled:

```
blind_spot_confirm_tool --slog-dir chunk/ --leaf-model <leaf_eval .onnx> \
  --leave-inference <leave_inference .onnx> --gcg-dir found/
```

- **The screen is the pool.** A pool sims every legal move, so a pool whose
  leader (the largest `q`) is outside the static-equity top `--cut` is a
  candidate.
- **The confirmation** re-sims the cut's moves and the pool's best `--picks`
  outside moves at `--rollouts` each (5000), unraced, on rollouts independent
  of the screen's, under the leaf network and leave inference model the
  pool's header names; the tool refuses others. A position is kept when the
  best outside move beats the best cut move by `--min-gain` points of win
  percentage (5) and by `--sigmas` paired standard errors (2).
- **Output:** `<stem>.blindspots.json` beside each `.slog`, holding every
  confirmed position, kept or not, and the number of pools screened; and a
  `.gcg` per kept position in `--gcg-dir`, named as the workload's are. The
  survey viewer does not read this JSON: the GPU returns only each rollout's
  outcome and its two moves, not the survey's rollout summaries.

**From the dashboard**, the workload **Collect hidden-leaves blind spots
(GPU)** (`hidden_blind_spots`) runs the whole chain per cycle: a 200-game
chunk of hidden-leaves HastyBot self-play, its `.forgone` sidecar, its pools
(`policy_set_pool_tool`, as `policy_set`'s generators label them), then the
confirmation. Create a tag naming the leaf eval and leave inference
generations (as for a `policy_set` tag; -1 pins the latest export), add a
local **Generator (GPU)** worker, and start it. The controller finishes the
generators once the tag holds `target_positions` kept positions (1000 by
default). The tools are not in the worker bundle, so the role runs on local
workers only. Each chunk delivers into `/workspace/mount/tags/hidden_blind_spots/<tag>/data/`:

- `found/<stem>.slog`, `found/<stem>.forgone`: the chunk's games and their
  forgone plays, which step 4's tools replay; the pools are dropped;
- `found/<stem>.blindspots.json`: the confirmation's report;
- `gcg/<stem>-g<G>-turn<N>.gcg`: each kept position.

On the 2,000 pools of a `tilted2`-gen-1188 / `leaf-conv-join-32-4b`-gen-730
chunk set (2026-10-25), 238 pools had a leader outside the top 10, 51 of those
gained at least 2 points on confirmation, and 28 at least 5: one blind spot
per 71 pools, in 95 seconds on the dev laptop's GPU. 22 of the 28 had 7 or
fewer tiles in the bag, where the rollouts play out to the end by HastyBot
rather than stop at the leaf network.
