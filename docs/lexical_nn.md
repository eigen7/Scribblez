# Max-move-per-lane: a lexical probe

## Goal

Train a network whose weights encode knowledge of a Scrabble lexicon, through
a probe task: given a board and a pre-move rack, predict the highest-scoring
move and its score. Training data is HastyBot-vs-HastyBot self-play (`.slog`;
every turn, endgame included, is a training row) under the generational loop
([generational_training.md](generational_training.md)). It runs as the
`max_move_per_lane` dashboard workload.

This is the largest of the lexical-NN experiments; the smaller ones are
[word_validity_experiments.md](word_validity_experiments.md) and
[rack_best_experiments.md](rack_best_experiments.md).

## Sub-tasks: one per lane

The problem decomposes into **30 sub-tasks**: the maximal-scoring move in each
of the 15 rows (best horizontal play) and 15 columns (best vertical play), so
each sample teaches several "best word here" facts. The decomposition is a
clean partition:

- A multi-tile play belongs to exactly one lane, the one it lies along.
  Incidental cross-words do not make it eligible for perpendicular lanes.
- A **single-tile play** has no direction of its own. It belongs to a lane iff
  it forms a word along that lane, so a tile that forms words both ways lands
  in both its row and its column, each credited with the play's full score.

30 lanes were chosen over the finer 450 per-(square, direction) decomposition
because the target stays small and a lane's union is *recoverable*: when a
lane's max move is unique, the union tensor is exactly that word's footprint,
so the move can be read back off the output. The globally best move is a
structural max over the 30 per-lane scores, with no separate loss.

## Targets

- **Move (union) target.** Per lane, a `(27, 15)` tensor: entry `(x, y)` is 1
  iff some lane-maximal move places tile kind `x` on the lane's `y`th square.
  The 27 kinds are the 26 letters plus one for a blank, whatever letter it
  designates. Only newly placed tiles are marked. Masked BCE; a lane with no
  legal move is fully masked.
- **Score target.** Per lane, 100 score bins (the last is "≥ 99"), with a PDF
  loss (cross-entropy) and a CDF loss (discrete CRPS). Masked for no-move
  lanes.
- **Has-move.** Per lane, BCE over all 30 lanes. It gates the structural max,
  so an empty lane cannot win it.

The C++ side of the layout is `engine/include/training/lane_targets.h`.

## Input encoding

The task has its own lean encoder, deliberately not the position-eval one. The
position-eval encoder's cross-check planes *are* lexicon knowledge, and
feeding them to a network whose purpose is to learn the lexicon would defeat
the experiment. Score differential, unseen pool and move history are
irrelevant and dropped. The input is letter planes, a blank marker,
premium-square planes, and the raw 27-entry rack counts (exact counts matter:
"can I play two R's" is a counting fact).

## Architecture

`py/scribblez/max_move_per_lane/model.py` (`MaxMovePerLaneModel`) splits the
work into "where" and "what word":

- **Spatial stage.** The `SpatialTrunk` shared with the position evaluation
  model encodes the board into per-cell features. Convolution carries the
  spatial facts: premium geometry, tile placement, openness.
- **Lexical stage.** One small transformer encoder runs along every lane
  (rows and columns, with the same weights) over the trunk's per-cell
  features plus prepended rack tokens. A transformer carries the lexicon,
  rather than more convolution, for two reasons. A word threads *through*
  existing tiles, so its letters are non-adjacent in the lane, and
  self-attention binds them in one layer. And under the key-value-memory view
  of transformers, FFN width is lexical capacity. Sharing weights across axes
  makes main-word scoring and cross-word checking the same operation.

The heads, also shared across axes, are per-cell occupancy (the union target),
per-lane score bins from the pooled lane vector, and per-lane has-move.

An optional frozen lexicon tool (the `lexicon_module` param;
[lexical_tools.md](lexical_tools.md)) can feed the lane transformer and
replace part of its FFN capacity.

## Dashboard

The workload's tabs are Loss, Lane analysis, Controls and Info
([react_dashboard.md](react_dashboard.md)). The Loss tab's panels
auto-discover every `loss_<x>` and `<x>_acc` series, so a new auxiliary loss
term needs no schema or front-end change. The accuracy metrics are exact
union match (`move_acc`), argmax score-bin match (`score_acc`), and has-move
accuracy (`has_move_acc`).
