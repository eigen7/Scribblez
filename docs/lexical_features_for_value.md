# Lexical foresight via engineered features (for the value models)

## Purpose

The value models (the position evaluation model and the move set evaluation model; see [roadmap.md](roadmap.md)) need
**board-conditional leave evaluation with lexical foresight**: the value of the
tiles a player keeps depends on what those tiles can *do* on this specific board,
which is a lexical fact. Rather than teach the network the lexicon, the plan is
to compute the relevant lexical facts with the classical GADDAG move generator
and feed them to the network as input features — the network then only has to
learn to *value* them, not to derive them.

This document records the motivation, why deep search (Monte Carlo) alone is
insufficient, why that leads to lexical *input features* rather than lexical
*network internals*, and the concrete feature designs together with their
tradeoffs and the optimizations that keep them affordable.

## Where this sits in the system

- **The position evaluation model** evaluates a position after the active player places tiles but
  before drawing replacements. It already sees the post-play **leave**, so
  evaluating the leave is squarely its job.
- **The move set evaluation model** predicts what the position evaluation model would say for each of `N` candidate moves in
  a single cross-attention pass (the board is encoded once and each move attends
  into it, so scoring is `O(N)` — no candidate-set reduction is needed before
  scoring). It acts as a **filter**: the top `K` moves by predicted value reach
  Monte Carlo rollouts (see roadmap Phase 4 pipeline).
- **Move generation is classical GADDAG.** The network is never asked to *find*
  legal words — it is handed only legal candidates and asked for their *value*.
- **Monte Carlo** does the deep, honest evaluation of the few survivors.

One of the two weaknesses the roadmap exists to fix is **"context-blind leave
evaluation."** The features described here are a direct mechanism for it.

## Motivation: context-dependent leave value

Leave value is board-dependent, and the dependence is lexical. Concrete example:

> A column is open such that `ZEIN` would score ~50 (the `Z` lands on a
> double-letter square and the word reaches a double-word square). The player
> holds `Z`, `I`, `N` but **not** `E`, so `ZEIN` is not playable this turn. The
> strong play may be to play off four *other* tiles elsewhere and **keep the
> `ZIN` leave**, hoping to draw an `E` next turn and land `ZEIN`.

For an engine to choose this, the value it assigns to the four-tile play must
reflect the `ZIN` leave's board-conditional upside. A static, board-independent
leave table (what HastyBot's static equity uses) cannot: it values `ZIN`
identically on every board. A context-aware value net is supposed to fix this —
**but only if it can "see" that `ZEIN` is reachable here**, which is a lexical
fact about this board combined with the retained tiles.

### Why the network cannot be trusted to derive this itself

The lexical-NN probe track ([lexical_tools.md](lexical_tools.md),
[lexical_nn.md](lexical_nn.md), [word_validity_experiments.md](word_validity_experiments.md),
[rack_best_experiments.md](rack_best_experiments.md)) investigated whether a
network can internalize or query lexical structure. Two findings matter here:

1. A frozen lexicon *tool* helps only when its structure matches the task's
   shape (e.g. an ordered DAWG walk wins word-validation but does nothing on an
   unordered rack).
2. On the board-anchored max-move-per-lane task, a plain network reliably
   recovers a play's *score* and its high-value anchor tiles (which follow from
   premium-square geometry) but **fails to fill the interior letters** of the
   word — precisely the part that requires computing dictionary membership.

That negative result is the justification for this whole approach: if the
network cannot cheaply derive lexical reachability from raw board + rack, then
compute it externally and hand it over.

## Why "let Monte Carlo figure it out" is insufficient

A natural objection: Monte Carlo rollouts of the four-tile play would reveal the
`ZIN` leave's quality, so why engineer a feature?

Because **Monte Carlo only runs on candidates that survive the move-set-evaluation filter.**
If the pre-move model cannot value the four-tile play's leave upside, that play
falls outside the top-`K` and never reaches simulation. The lexical signal
therefore has to live **upstream, in the features the move set evaluation model sees** — not only in
the deep search.

Framed as a recall/precision split:

- The **feature's job is recall**: ensure lexically promising leaves survive the
  filter and reach simulation.
- **Monte Carlo's job is precision**: honestly evaluate the survivors.

This split is what makes aggressive feature approximations acceptable (below): a
feature need not be *accurate*, only *informative enough to prevent the filter
from discarding the play*.

## Why features, not network lexical internals

The lexical query that matters — "the best move achievable with rack =
`leave ∪ {X}` on this board" — is well defined and the classical GADDAG computes
it in microseconds. It requires **no differentiability**.

The lexical-NN track's hard part was making lexical lookup *differentiable* and
*learnable to query* (soft top-K DAWG state, straight-through estimators, etc.).
A feature skips all of that: compute the answer with the fast classical
generator, feed it in, let the network map features → value. Every headwind the
probe track hit (wrong-shaped tool, one-shot per-cell loss that cannot spell, no
rack awareness, board-anchoring needing a GADDAG) simply does not arise. This
also matches the roadmap's division of labor: **the classical engine does the
lexical work; the network does the value.**

## The cost/accuracy ladder

There are three tiers for computing a "contingent draw" feature, differing in
which board and which rack the hypothetical generation runs against:

1. **Shared, full-rack, current board** — computed once per position from the
   current rack, shared across all candidate moves. Cheapest. Most approximate:
   uses the full rack (not any specific leave), and is blind to how the candidate
   move changes the board (both self-block and self-created opportunities).
2. **Per-leave, current board** — computed per distinct leave. Moderate cost: a
   leave is a rack-subset, so the position has at most a couple hundred distinct
   leaves regardless of how many moves share each. Correct about
   leave-sufficiency; still blind to self-block and self-created opportunities.
3. **Per-move, post-move board** — the technically correct version: the
   contingent play is generated on the board *as it will be after the candidate
   move*, so it accounts for the move blocking (or opening) the very lane the
   leave would exploit. **Prohibitive** — it is a full contingent generation per
   candidate move.

Tier 3 is the reason a naive "just do it right" approach fails on cost. Tier 1
is acceptable *because* of the recall/precision split: an occasional over-credit
of a leave whose contingent play the move actually blocks is caught later, when
the surviving move is honestly evaluated by Monte Carlo.

## The 27×30 potential map

The core shared feature (tier 1):

> For each drawable tile `X` (26 letters + blank) and each of the 30 lanes
> (15 rows + 15 columns), the best move the rack `∪ {X}` can make along that
> lane, **restricted to moves that use `X`**.

Two structural points make this affordable and useful:

- **Per-lane is free bookkeeping.** Finding the single best move for `rack ∪ {X}`
  (the "27×1" version) already requires enumerating every lane and taking the
  board-wise max. Retaining the per-lane bests (the "27×30" version) costs no
  extra *generation* — only storage/encoding.
- **Encode letters + cells + score per entry**, not just a score. The network
  must be able to check two things:
  1. **Leave-sufficiency** — do the retained tiles (plus the drawn `X`) actually
     supply the letters the contingent play needs?
  2. **Geometric conflict** — does the candidate move occupy any of the cells the
     contingent play needs?

  Both checks require the placed letters and cells to be in the feature.

Weighting the entries by the bag's remaining draw probabilities gives a compact
summary (an expected contingent score) that may capture most of the value on its
own.

### Limitations of the shared map (read these — they fail silently)

- **Full-rack ≠ leave (optimism).** The best `rack ∪ {X}` move along a lane may
  use tiles the player will *not* keep, so it need not be compatible with any
  particular leave. Example: the best D-lane play with `AAEINRZ + E` might be a
  six-tile word using both `A`s — irrelevant to a `ZIN` leave — while the
  leave-compatible `ZEIN` is only the second-best move and is never encoded.
  Encoding only the per-`(tile, lane)` **maximum** therefore can hide exactly the
  contingency that motivated the feature, and it does so *silently*.
  Mitigations: keep **top-k per `(tile, lane)`**, or bias the retained move toward
  ones that use a scarce high-value held tile (e.g. require it uses the `Z`).
- **Blind to self-block and self-created opportunities** — inherent to computing
  on the current board (tiers 1–2). Self-created opportunities are addressed by a
  separate feature (below).

## Optimizations to avoid a 27× blowup

- **Restrict to moves that use the added tile.** This prunes the enumeration
  heavily — most `rack ∪ {X}` moves do not place `X` and are irrelevant.
- **Lanes are independent** → the per-lane generations parallelize.
- **27×30 is not more generation than 27×1** (see above): the board-wise max
  already enumerates all lanes.
- **Possible single augmented pass (worth prototyping).** GADDAG generation
  already tracks rack consumption as it walks anchors. It may be possible to run
  *one* enriched generation per lane that treats the 27 possible bonus tiles as
  optional extra rack slots and records, per bonus letter, the best move that
  *consumed* that slot — collapsing 27 passes into roughly one. Feasibility
  depends on the movegen internals (the anchor / cross-set walk); prototype
  before accepting a 27× constant factor.
- **Amortize and cache.** Precompute at self-play data-generation time and store
  alongside the `.slog` data. Cache per `(board, rack/leave)`. Restrict `X` to
  tiles actually remaining in the bag.

## Newly created opportunities (the hook subtlety)

A move's value includes the board structure it **creates** for the leave to
exploit next turn — e.g. playing a word that opens an `S`-hook on a triple-word
lane when the player kept an `S`. The current-board contingent features
(tiers 1–2) miss this entirely, because the hook does not exist until the move is
made.

**Feature: the post-move cross-check delta.** Encode the *changes* a candidate
move makes to the board's cross-check sets (which letters are legal at each empty
square along the perpendicular axis), rather than the full cross-check planes.

- **The delta is sparse.** A move changes cross-sets only in the squares
  perpendicular-adjacent to its placed tiles and at its lane's extension points —
  `O(tiles placed)` squares, each a 26-bit cross-set per axis. Encoding that
  handful of `(square, axis, changed-letters)` entries is far smaller than the
  full `26×15×15×2` cross-check representation, and it belongs in the move set evaluation model's
  **per-move** embedding (it is a property of the specific move).
- **Complementary to the potential map.** The 27×30 map captures exploiting
  *existing* board structure; the cross-check delta captures exploiting structure
  the move *creates*. Neither subsumes the other, and they attach to different
  halves of the move set evaluation model (shared board features vs. the per-move embedding).

## Mapping onto the move set evaluation model

- The **shared potential map** becomes additional board-encoder features that
  each move embedding attends over via the move set evaluation model's cross-attention. Geometric
  conflict between a candidate move and a contingent play is spatial overlap,
  which attention/convolution detect well.
- The **cross-check delta** rides in the per-move embedding.

## Risks and recommended sequencing

- **Nothing here is measurable without an evaluation bank.** This is a
  substantial C++ movegen + input-encoder effort whose payoff is invisible until
  it can be tested. Build a small probe bank of contingent-leave positions (the
  `ZIN`/`ZEIN` case is the seed) *and* hook-creation positions, and check whether
  the value model ranks the leave-preserving / hook-opening play correctly. This
  doubles as a Phase 3 position bank, so it is not throwaway work.
- **Cheap before rich.** Start with a bag-probability-weighted
  expected-contingent-score scalar (per lane or per position) plus the
  cross-check delta; ablate on the bank; escalate to the full 27×30-with-encoded-
  moves map (and top-k per entry) only if the cheap version shows signal but
  plateaus. The rich map is significant compute *and* significant relational
  reasoning for the network to learn — measure its marginal value over the scalar
  before building it.

## Pointers

- [contingent_map.h](../engine/include/scribblez/contingent_map.h) — the
  position evaluation model implementation of the potential map. For the position evaluation model the cost/accuracy
  ladder collapses: the input state is already post-move (rack = leave, board
  includes the move), so one generation per position yields the tier-3-correct
  feature. The 27 per-tile passes collapse into a single generation over
  rack ∪ {blank} (a play consuming the extra blank designated as `L`,
  rescored at `L`'s face value, is the "drew `L`" play). Encoded as input
  planes 85–87 plus 56 scalars (see
  [input_encoder.h](../engine/include/scribblez/input_encoder.h)).
- [roadmap.md](roadmap.md) — the position evaluation model and the move set evaluation model, the candidate-scoring/selection
  pipeline, the two weaknesses.
- [architecture.md](architecture.md) — the input encoder and the `.slog`
  data-generation pipeline (where precomputed features would be stored).
- [lexical_tools.md](lexical_tools.md), [lexical_nn.md](lexical_nn.md),
  [word_validity_experiments.md](word_validity_experiments.md),
  [rack_best_experiments.md](rack_best_experiments.md) — the lexical-NN probe
  track and the findings that justify computing lexical facts externally.
