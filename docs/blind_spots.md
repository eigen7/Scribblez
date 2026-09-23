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
   ./py/scripts/blind_spots_collect.py --tag <tag> [--min-gain 2]
   git add positions/NWL23/best-bot-blind-spots
   ```

   `blind_spots_collect.py` rebuilds the directory from the tag each time, so
   rerun it as the tag grows. `--min-gain` drops plays whose edge over the
   best top move is under that many percentage points of win rate. It is
   needed because, with 5000-rollout confirming sims, the survey's own
   two-standard-error bar admits edges well under one point.

## What a worker delivers

Into `/workspace/mount/tags/blind_spots/<tag>/data/`, by whichever route the
slot uses (a rename for a local worker, over ssh from an operator's machine,
through the bucket from a rented one):

- `survey/<stem>.simsurvey.json`: one per game, slimmed to the positions
  found. Each position lists the top moves and the outside picks with their
  confirming-sim summaries (`engine/include/sim/rollout_summary.h`). The file
  also records `positions_surveyed`, the number of turns surveyed, so rates
  stay computable.
- `gcg/<stem>-g0-turn<N>.gcg`: the game up to each found position.

`<stem>` is `<timestamp>-<worker id>`, unique across workers. A game's full
survey (every legal play's screening summary, megabytes per game) never leaves
the worker and is discarded. To keep full surveys, run
`py/scripts/sim_candidate_survey.py` on one machine instead.
