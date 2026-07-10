# Lexical foresight via engineered features (for the value models)

## Purpose

The value models need **board-conditional leave evaluation with lexical
foresight**: the value of the tiles a player keeps depends on what those tiles
can *do* on this specific board, which is a lexical fact. Rather than teach
the network the lexicon, compute the relevant lexical facts with the classical
GADDAG move generator and feed them in as input features — the network then
only has to learn to *value* them, not derive them.

The seed example: `ZEIN` would score ~50 in an open column, the player holds
`Z`, `I`, `N` but no `E` — the strong play may be to play four *other* tiles
and keep the `ZIN` leave, hoping to draw the `E`. A board-independent leave
table values `ZIN` identically on every board; a context-aware net fixes this
only if it can *see* that `ZEIN` is reachable here.

The network cannot be trusted to derive this itself: the lexical-NN probe
track ([lexical_nn.md](lexical_nn.md),
[word_validity_experiments.md](word_validity_experiments.md),
[rack_best_experiments.md](rack_best_experiments.md)) showed that a plain
network recovers a play's score and anchor geometry but fails to fill the
interior letters — precisely the part requiring dictionary membership — and
that lexicon *tools* help only when their structure matches the task's shape.
And "let Monte Carlo figure it out" is insufficient because **Monte Carlo only
runs on candidates that survive the move-set-evaluation filter**: if the
pre-move model cannot value the leave upside, the play never reaches
simulation.

The recall/precision split follows: the **feature's job is recall** (get
lexically promising plays past the filter), **Monte Carlo's job is precision**
(honestly evaluate the survivors). This is what makes aggressive feature
approximations acceptable — a feature need not be accurate, only informative
enough to prevent the filter from discarding the play.

The lexical query that matters ("best move achievable with `leave ∪ {X}` on
this board") is exactly what the GADDAG computes in microseconds, with no
differentiability needed — the classical engine does the lexical work, the
network does the value.

## The cost/accuracy ladder

Three tiers for a "contingent draw" feature, by which board and rack the
hypothetical generation runs against:

1. **Shared, full-rack, current board** — once per position, shared across
   candidates. Cheapest; blind to leave-specificity and to how the candidate
   changes the board.
2. **Per-leave, current board** — per distinct leave (a position has at most
   a couple hundred). Correct about leave-sufficiency; still blind to
   self-block and self-created opportunities.
3. **Per-move, post-move board** — technically correct, **prohibitive**: a
   full contingent generation per candidate.

Tier 1 is acceptable *because* of the recall/precision split: an occasional
over-credit is caught by the sims downstream.

## The 27×30 potential map

The core shared feature (tier 1): for each drawable tile `X` (26 letters +
blank) and each of the 30 lanes, the best move `rack ∪ {X}` can make along
that lane, **restricted to moves that use `X`**. Two structural points:

- **Per-lane is free bookkeeping**: finding the single best move already
  enumerates every lane, so retaining per-lane bests costs only encoding.
- **Encode letters + cells + score per entry**, not just a score: the network
  must check *leave-sufficiency* (do the retained tiles supply the letters?)
  and *geometric conflict* (does the candidate occupy the contingent play's
  cells?), and both need the placed letters and cells present.

Weighting entries by the bag's draw probabilities gives a compact
expected-contingent-score summary.

**Silent-failure limitation**: the best `rack ∪ {X}` move along a lane may
use tiles the player will not keep, so the per-`(tile, lane)` *maximum* can
hide exactly the leave-compatible contingency that motivated the feature
(the six-tile play using both `A`s outranks `ZEIN`). Mitigations: top-k per
`(tile, lane)`, or bias retention toward moves that consume a scarce
high-value held tile.

Cost controls: restrict to moves using the added tile (prunes heavily); lanes
parallelize; possibly one enriched generation per lane treating the 27 bonus
tiles as optional rack slots (prototype against the movegen internals before
accepting a 27× constant); precompute at data-generation time and restrict
`X` to tiles remaining in the bag.

## Newly created opportunities: the post-move cross-check delta

A move's value includes board structure it *creates* for the leave to exploit
(opening an `S`-hook on a triple lane while keeping an `S`) — invisible to
current-board features. The feature is the **delta** a candidate move makes
to the board's cross-check sets: sparse (`O(tiles placed)` squares, one
cross-set per axis), and a property of the specific move, so it rides in the
move set evaluation model's **per-move** embedding. Complementary to the
potential map (existing structure vs created structure); the map attaches to
the shared board encoding, the delta to the per-move half.

## Risks and sequencing

- **Nothing here is measurable without an evaluation bank.** Build a small
  probe bank of contingent-leave and hook-creation positions and check
  whether the value model ranks the leave-preserving / hook-opening play
  correctly; it doubles as a Phase 3 position bank.
- **Cheap before rich.** Start with the expected-contingent-score scalar plus
  the cross-check delta; escalate to the full 27×30-with-encoded-moves map
  (and top-k entries) only if the cheap version shows signal but plateaus —
  the rich map is significant compute *and* significant relational reasoning
  to learn.

## Pointers

- [contingent_map.h](../engine/include/encoding/contingent_map.h) — the
  implemented potential map on the position evaluation model, where the
  ladder collapses: the input state is already post-move (rack = leave, board
  includes the move), so one generation per position yields the
  tier-3-correct feature, and the 27 per-tile passes collapse into a single
  generation over `rack ∪ {blank}` (a play consuming the extra blank
  designated `L`, rescored at `L`'s face value, is the "drew `L`" play).
  Encoding in [input_encoder.h](../engine/include/encoding/input_encoder.h).
- [roadmap.md](roadmap.md) — the value models and the candidate-selection
  pipeline; [architecture.md](architecture.md) — the data pipeline where
  precomputed features are stored.
- [lexical_nn.md](lexical_nn.md) and the experiment docs — the probe-track
  findings that justify computing lexical facts externally.
