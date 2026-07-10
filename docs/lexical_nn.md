# Lexical Neural Network

## Goal

Train a NN whose weights encode knowledge of a Scrabble lexicon, via a probe
task: given a pre-move rack and a board, predict the highest-scoring move and
its point value. Training data is hasty-vs-hasty self-play (`.slog`, every
turn a training row, endgame included) under the generational loop.

## Sub-tasks: one per lane

The problem decomposes into **30 sub-tasks**: the maximal-scoring move in each
of the 15 rows (best horizontal play) and 15 columns (best vertical play), so
each sample teaches several "best word here" facts. This is a clean
partition: every multi-tile play belongs to exactly one lane (its lay
direction); incidental cross-words do not make it eligible for perpendicular
lanes. **Single-tile plays** have no lay direction and are classified into a
lane iff they form a word in that direction (a true cross lands in both
lanes), with the play's full score attributed to whichever lane(s) it lands
in.

30 lanes was chosen over the finer 450 per-(square, direction) decomposition
because the target stays small and a lane's union is *recoverable*: with a
unique max move, the union tensor is exactly that word's footprint, so the
consumer can read the move back off the output. The globally best move is a
structural max over the 30 per-lane scores (no separate loss).

## Targets

- **Move (union) target**: per lane, a `(27, 15)` tensor — entry `(x, y)` is
  1 iff some lane-maximal move places tile `x` on the lane's `y`th square
  (26 letters + 1 uninstantiated blank; only newly placed tiles marked).
  Masked BCE; a lane with no legal move is fully masked.
- **Score target**: per lane, 100 score bins (last bin = "≥ 99"), with a
  PDF loss (cross-entropy) and a CDF loss (discrete CRPS). Masked for
  no-move lanes.
- **Has-move**: per lane, BCE over all 30 lanes; gates the structural max so
  empty lanes can't win it.

## Input encoding

The task has its own lean encoder, deliberately *not* the position-eval one:
the position-eval encoder's cross-check planes **are lexicon knowledge**, and
feeding them to a net whose purpose is to learn the lexicon defeats the
experiment. Score differential, unseen pool, and move history are irrelevant
and dropped. Sufficient input: letter planes + blank marker + premium-square
planes + raw 27-entry rack counts (exact counts matter — "can I play two
R's" is a counting fact).

## Architecture

`py/scribblez/max_move_per_lane/model.py` (`MaxMovePerLaneModel`), split into
"where" and "what word":

- **Spatial stage**: the `SpatialTrunk` shared with the position evaluation
  model encodes the board into per-cell features — convolution carries the
  spatial facts (premium geometry, tile placement, openness).
- **Lexical stage**: one small transformer encoder run along every lane (rows
  and columns with transpose-shared weights) over the trunk's per-cell
  features plus prepended rack tokens. A transformer, not more convolution,
  carries the lexicon because a word threads *through* existing tiles — its
  letters are non-adjacent in the lane, and self-attention binds them in one
  layer — and because FFN width is lexical capacity under the
  key-value-memory view. Shared weights across axes make main-word scoring
  and cross-word checking the same operation on two axes.

Heads (shared across axes): per-cell occupancy → the union target; per-lane
score bins from the pooled lane vector; per-lane has-move.

## Dashboard

The trainer shares the per-tag dashboard with the position-eval trainer; the
loss and accuracy panels auto-discover every `loss_<x>` / `<x>_acc` series,
so auxiliary loss terms need no schema or front-end change. Accuracy metrics:
exact union match, argmax score bin match, and has-move accuracy.
