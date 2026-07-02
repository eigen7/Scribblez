# Large post-move-value Monte-Carlo test set

A machine-harvested companion to `../post-move-value-test-dataset/` (10 hand-built
positions), scaled up for gauging the impact of features on the post-move-value
model. Every position is a **penultimate-bingo** board: the opponent bingoed on
the previous turn, so its rack is a clean full draw and a Monte-Carlo rollout can
sample it uniformly from the unseen pool with no opponent-leave model. See
`docs/lexical_features_for_value.md` for why this set matters and its caveats
(it measures the penultimate-bingo slice, not the full position distribution).

## Contents

- `part-NNN.gcgs` — the harvested positions, ~100 concatenated GCG blocks per
  file. Each block begins with `#character-encoding UTF-8` (the record boundary)
  and keeps every move line's `rack_before` (the sim reads the final mover's
  leave from it) but contains **no** `#Rack` pragmas, so the bingoer's actual
  drawn rack is never recorded. A `#note` on each block carries its source game
  seed. These are committed.
- `monte-carlo-sim-results.json` — the MC ground truth (W/L/D + exact
  score-delta histogram) keyed by position stem. Committed.
- `pos-*.gcg` — transient loose files the C++ tools consume; produced by
  exploding the bundles and removed after scoring. Git-ignored, never committed.

## Regenerating / extending

Built by `py/scripts/build_post_move_test_set.py` (harvest → bundle → explode →
Monte-Carlo score). Harvested games use a reserved base seed (default 1000000)
disjoint from training self-play, so this set is never a training-contamination
risk. Re-score the committed bundles without replaying games via `--skip-harvest`.
Extend by harvesting more positions (more `part-*.gcgs` files) and re-scoring.
