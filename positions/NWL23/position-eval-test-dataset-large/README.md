# Large position-evaluation Monte-Carlo test set

A machine-harvested companion to `../position-eval-test-dataset/` (the
hand-built positions), scaled up for gauging the impact of features on the
position evaluation model: post-move positions sampled from HastyBot self-play
(`harvest_positions_tool`: one training-eligible tile placement per game,
uniformly), each with a Monte-Carlo ground truth. The dataset contract is the
small set's (see its README). See `docs/lexical_features_for_value.md` for
why this set matters and its caveats.

## Contents

- `part-NNN.gcgs` — the harvested positions, ~100 concatenated GCG blocks per
  file. Each block begins with `#character-encoding UTF-8` (the record boundary)
  and keeps every move line's `rack_before` (the sim reads the final mover's
  leave, and the opponent's retained leave, from them) but contains **no**
  `#Rack` pragmas, so neither player's post-move draw is recorded. A `#note`
  on each block carries its source game seed. These are committed.
- `monte-carlo-sim-results.<condition>.json` — the MC ground truth (W/L/D +
  exact score-delta histogram) keyed by position stem, one file per
  information condition (what a rollout knows of the opponent's leave;
  `engine/include/sim/monte_carlo_sim.h`). Committed, always two real files
  (never a symlink: the scoring tool writes each condition's file, and a link
  would route one condition's write into the other's file).
- `pos-*.gcg` — transient loose files the C++ tools consume; produced by
  exploding the bundles and removed after scoring. Git-ignored, never committed.

## Regenerating / extending

Built by `py/scripts/build_position_eval_test_set.py` (harvest → bundle → explode →
Monte-Carlo score). Harvested games use a reserved base seed (default 1000000)
disjoint from training self-play, so this set is never a training-contamination
risk. Re-score the committed bundles without replaying games via `--skip-harvest`.
Extend by harvesting more positions (more `part-*.gcgs` files) and re-scoring.
