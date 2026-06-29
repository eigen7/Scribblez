# Lexical Neural Network

## Goal

Train a NN whose weights encode knowledge of a Scrabble lexicon.

Existing NN training pipelines like `py/scripts/train_post_move_model.py` train a CNN, which
likely lacks the capability to learn the thousands of commonly occurring Scrabble words.

## Learning Setup

Similarly to `train_post_move_model.py`, play fast hasty-vs-hasty games in c++ and stream training
data, one-sample-per-game, to the python side.

Unlike `train_post_move_model.py`, we will sample any position from the game, including endgame
positions (sample uniformly over all turns).

The learning problem is to take a pre-move rack and a board, and to predict the highest-scoring
move, along with the point-value of that move. The expectation is that this task will require the NN
to encode the lexicon in its weights.

### Sub-tasks: one per lane

Rather than learn only the globally maximal move, we decompose the problem into **30 sub-tasks**:
the maximal-scoring move in each of the **15 rows** (best horizontal play in that row) and **15
columns** (best vertical play in that column). Each sub-task contributes to the loss, so each sample
teaches the net several "what is the best word here" facts, not just one.

This is a clean partition: every multi-tile play belongs to exactly one lane (the row or column its
tiles lie along — its *lay direction*), so a lane's target is a single best move (plus any ties).
Its incidental cross-words do **not** make it eligible for perpendicular lanes.

A 30-lane decomposition is chosen over a finer per-(starting-square, direction) one (which would be
15x15x2 = 450 sub-tasks) for two reasons: the target stays small and, more importantly, a lane's
union is *recoverable* — with a unique max move, the union tensor (below) is exactly that word's
placed-tile footprint, so the consumer can read the move back off the output. The cost is less
lexical signal per sample than 450 sub-tasks would give; finer granularity can be reintroduced later
if the net saturates.

**Single-tile plays** have no lay direction. They are classified into a lane only if they *score in
that direction*: into the row sub-task iff they form a horizontal word, into the column sub-task iff
they form a vertical word. A true cross (a word in both directions) lands in both lanes; an
isolated-axis tile lands in the one direction it forms a word. The play's full score (main +
cross-word points) is attributed identically to whichever lane(s) it lands in — it is one physical
move with one score.

The globally highest-scoring move is the max over all 30 sub-tasks. The net exposes this via a
max-pooling output head over the 30 per-lane score predictions; supervision is per-lane only and the
global max is purely structural (no separate loss on it).

## Encoding a Move

In general, multiple moves can be tied for highest scoring in a lane. We frame each sub-task as
learning the **union** of all moves that are maximal *for that lane*. A lane's union is a (27, 15)
tensor: entry (x, y) is 1 iff some maximal move in the lane places tile x on the y'th space of the
row/column. 27 = 26 letters + 1 (all blank instantiations collapsed to a single uninstantiated
blank). Stacking the 30 lanes gives a (30, 27, 15) ~= 12k-entry target. Equivalently, this is two
(15, 15, 27) per-cell tensors (one for the row sub-tasks, one for the columns): cell (r, c) of the
row tensor holds the tile that row r's best horizontal move places at column c.

Only newly-placed tiles are marked (tiles the move threads through are already on the board, which
the consumer has). Each entry contributes BCE loss, masked per-lane: a lane with no legal move
masks out all 15 of its cells.

Possible further compaction, if it yields meaningful gains: specify tiles as rack-indices instead of
letters (27 -> 7).

## Encoding a Score

Each lane has its own max-score prediction: 100 logit outputs B_0, B_1, ..., B_99, where B_k is an
indicator of "this lane's max-score is == k" for k < 99, and B_99 is an indicator of "max-score is
>= 99". A lane with no legal move is masked out of the score loss.

We have two loss terms associated with each lane's score: a PDF-loss, and a CDF-loss.

For PDF-loss, we can use sum_{0 <= k <= 99} B(k) log(B_hat(k)).

For CDF-loss, we can use sum_{0 <= k <= 99} (sum_{j <= k} B(j) - B_hat(j))^2.

## Input Encoding

This task needs its own lean input encoder, **not** the one from `train_post_move_model.py`. Two
points:

- The post-move encoder's spatial planes include horizontal/vertical cross-checks (which letters are
  legal at each empty square) — that is lexicon knowledge, and feeding it to a net whose purpose is
  to *learn* the lexicon defeats the experiment. Cross-checks must be excluded from the input. (They
  may instead become an auxiliary *output*/loss term; see Dashboard.)
- Score differential, unseen-tile pool, and last-move metadata are irrelevant to a single-move
  lexical task and should be dropped.

A sufficient input is roughly: 26 letter planes + 1 blank-marker plane + premium-square planes
(word/letter multipliers) + the 27-entry raw rack counts (exact counts matter — "can I play two
R's" is a counting fact).

## NN Architecture

The model is `py/scribblez/max_move_per_lane_model.py` (`MaxMovePerLaneModel`). It has two stages
that split the problem into "where" and "what word":

- **Spatial stage (CNN).** A conv trunk -- the `SpatialTrunk` shared with the post-move model (stem,
  rack-scalar injection, a residual tower with KataGo-style global-pooling blocks) -- encodes the
  board into a `(C, 15, 15)` feature map. Convolution is the natural tool for the spatial facts:
  premium-square geometry, which tiles sit where, board openness.

- **Lexical stage (the lexicon store).** A single small **transformer encoder is run along every
  lane** -- once per row and once per column, with *transpose-shared weights* -- over the trunk's
  per-cell features (plus a few rack tokens prepended so the lane can attend rack<->board). This is
  where the lexicon is learned. The key reasons a transformer, and not more convolution, carries the
  lexical knowledge: a word threads *through* tiles already on the board, so its letters are
  non-adjacent in the lane, and self-attention binds those disjoint positions in a single layer
  (a fixed-width conv cannot); and the dictionary itself lives in the FFN width under the key-value-
  memory view of a transformer -- attention indexes into it, width *is* lexical capacity. Running the
  same weights on rows and columns makes main-word scoring and perpendicular cross-word checking the
  same operation applied on two axes.

**Fusion** is simply that the lane transformer consumes the CNN's per-cell lane features: the conv
supplies spatially-grounded, premium-aware cell vectors, and the transformer turns each lane's
sequence of them into word-level judgments. Neither stage alone suffices -- conv can't bind a word's
non-adjacent letters, and a transformer on raw squares would have to relearn board geometry the conv
gives for free.

Heads, all shared across the two axes: a per-cell **occupancy** head (`C -> 27`) emits the
`(30, 15, 27)` union target; a per-lane **score** head reads the pooled (mean+max over cells) lane
vector and emits 100-bin score logits; a per-lane **has-move** head predicts whether the lane has any
legal play. The global best-move score is the structural max over the 30 lanes' expected scores,
gated by the has-move probability so empty (score-unsupervised) lanes can't win the max. Losses:
masked score-PDF (cross-entropy) + score-CDF (discrete CRPS), masked occupancy BCE, and has-move BCE
over all 30 lanes.

## Dashboard

We need a streaming dashboard similar to that produced by `train_post_move_model.py`. The key
metrics:

- Train loss (stacked plots: score-PDF, score-CDF, move). If any auxiliary loss terms are
added (cross-checks?), that should be another stack component.

- Train accuracy: for each of the 30 lane sub-tasks that permits at least one legal move: does it get
the move right? Does it get the score right? Show accuracy stats.

Having a database entry per-minibatch is too much, both for the sqlite3 tables and for the front-end
rendering. Maybe one per minute? Or maybe have one per-minibatch in the early going, so I can spot
any anomalies early-on, but then switch to  aggregating at some point, with the plots choosing the
resolution appropriately?

There is probably opportunity to reuse code from the existing dashboard. Please refactor
accordingly.
