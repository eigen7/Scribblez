# Lexical foresight via engineered features (for the value models)

**Status.** Two features were proposed. Neither is a model input today.

- **The contingent-draw potential map**: built as an optional input-encoding
  arm (3 planes plus 56 scalars on the position evaluation row), then
  deleted. No run adopted it, and carrying an unused arm through the
  encoder, the FFI, the ONNX metadata and every model config cost more than
  it bought. Anyone reviving it starts from this document and the deleted
  `engine/{include,src}/encoding/contingent_map.*` (removed in commit
  `d930b59`).
- **The post-move cross-check delta**: the encoder and a diagnostic have
  landed, not the model input. `training/cross_check_delta.h` computes each
  candidate's sparse cross-check change, and the FFI and
  `MsetDataset(with_cross_check_deltas=True)` expose it. The script
  `py/scripts/move_set_eval/crosscheck_delta_diagnostic.py` measures whether
  the student's distillation error concentrates where that delta is large.
  Feeding the delta into the move set evaluation model waits on that
  measurement.

**Goal.** Give the value models board-conditional leave evaluation with
lexical foresight, without asking the network to learn the lexicon.

**Decision.** Compute the relevant lexical facts with the classical GADDAG
move generator and feed them in as input features. The network then only
has to learn to *value* those facts, not derive them.

## Why

The value of the tiles a player keeps depends on what they can do on this
specific board, and that is a lexical fact. Example: `ZEIN` would score
about 50 in an open column, and the player holds `Z`, `I` and `N` but no
`E`. The strong play may be to play four *other* tiles and keep `ZIN`,
hoping to draw the `E`. A board-independent leave table values `ZIN` the
same on every board. A context-aware network fixes this only if it can see
that `ZEIN` is reachable here.

The network cannot be trusted to work this out itself. The lexical-NN probe
track ([lexical_nn.md](../lexical_nn.md),
[word_validity_experiments.md](../word_validity_experiments.md),
[rack_best_experiments.md](../rack_best_experiments.md)) found that a plain
network recovers a play's score and anchor geometry but fails to fill in the
interior letters, which is exactly the part that needs dictionary
membership. It also found that lexicon *tools* help only when their
structure matches the shape of the task.

Monte Carlo cannot cover for the network either, because it only runs on
candidates that survive the move set evaluation filter. If the pre-move model
cannot see a leave's upside, the play never reaches simulation.

That gives the division of labor: **the feature's job is recall** (get
lexically promising plays past the filter), and **Monte Carlo's job is
precision** (evaluate the survivors honestly). This is what makes aggressive
approximations acceptable. A feature does not need to be accurate, only
informative enough to keep the filter from discarding the play. And the
lexical query that matters, "the best move `leave ∪ {X}` can make on this
board", is what the GADDAG computes in microseconds, with no need to be
differentiable.

## The cost/accuracy ladder

A "contingent draw" feature can be computed at three tiers, by which board
and rack the hypothetical move generation runs against:

1. **Shared, full rack, current board.** Once per position, shared by every
   candidate. Cheapest; blind to which tiles a candidate keeps and to how
   the candidate changes the board.
2. **Per leave, current board.** Once per distinct leave (a position has at
   most a couple of hundred). Correct about whether the leave supplies the
   letters; still blind to a candidate blocking its own follow-up or creating
   a new one.
3. **Per move, post-move board.** Correct, and prohibitive: a full contingent
   generation per candidate.

Tier 1 is acceptable because of the recall/precision split: an occasional
over-credit is caught by the sims downstream.

## The 27×30 potential map

The core tier-1 feature: for each drawable tile `X` (26 letters plus the
blank) and each of the 30 lanes (15 rows, 15 columns), the best move
`rack ∪ {X}` can make along that lane, restricted to moves that use `X`.

- **Per-lane bests cost nothing extra to find.** Finding the single best
  move already enumerates every lane; keeping each lane's best costs only
  encoding.
- **Encode letters, cells and score per entry, not just a score.** The
  network has to check whether the retained tiles supply the letters and
  whether the candidate occupies the contingent play's cells. Both need the
  placed letters and cells.

Weighting entries by the bag's draw probabilities gives a compact
expected-contingent-score summary.

**Limitation: the maximum can hide the case that matters.** The best
`rack ∪ {X}` move along a lane may use tiles the player will not keep, so
the per-(tile, lane) maximum can hide exactly the leave-compatible play that
motivated the feature (a six-tile play using both `A`s outranks `ZEIN`).
Mitigations: keep the top k per (tile, lane), or bias retention toward moves
that consume a scarce high-value tile the player holds.

**Cost controls.** Restrict to moves that use the added tile (this prunes
heavily); parallelize over lanes; consider one enriched generation per lane
that treats the 27 bonus tiles as optional rack slots (prototype it against
the move generator's internals before accepting a 27× constant); precompute
at data-generation time; restrict `X` to tiles still in the bag.

**As implemented on the position evaluation model**, the ladder collapsed.
That model's input is already post-move (its rack is the leave, its board
includes the move), so one generation per position yields the tier-3 feature.
And the 27 per-tile passes became a single generation over
`rack ∪ {blank}`: a play that uses the extra blank as `L`, rescored at `L`'s
face value, is the "drew an `L`" play.

## Created opportunities: the post-move cross-check delta

Part of a move's value is board structure it *creates* for the leave to use,
such as opening an `S` hook on a triple lane while keeping an `S`. Features
of the current board cannot see it. The feature is the change a candidate
makes to the board's cross-check sets. It is sparse (at most two squares per
placed tile plus two for the word itself, one cross-set per axis) and a
property of the specific move, so it belongs in the move set evaluation
model's per-move embedding. It complements the potential map: the map
describes existing structure and attaches to the shared board encoding; the
delta describes created structure and attaches to the per-move half.

## Risks and sequencing

- **Nothing here can be judged without an evaluation bank.** Build a small
  bank of contingent-leave and hook-creation positions and check whether the
  value model ranks the leave-preserving or hook-opening play correctly.
- **Cheap before rich.** Start with the expected-contingent-score scalar
  plus the cross-check delta. Escalate to the full 27×30 map with encoded
  moves (and top-k entries) only if the cheap version shows signal and then
  plateaus: the rich map costs significant compute and asks the network for
  significant relational reasoning.

## Pointers

- [input_encoder.h](../../engine/include/encoding/input_encoder.h) owns the
  row layout a revived map would be encoded into.
- [roadmap.md](../roadmap.md) covers the value models and the
  candidate-selection pipeline; [architecture.md](../architecture.md) covers
  the data pipeline, where precomputed features would be stored.
