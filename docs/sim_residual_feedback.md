# Sim-residual feedback: re-evaluating moves with Monte Carlo evidence

## Purpose

Move selection (roadmap Phase 4) runs `M_pre` over all legal moves, sends the
top-`K` to Monte Carlo simulation, and picks the best simmed move. This document
proposes a **second evaluation round**: aggregate the simulation results,
compute the *difference* between what the sims observed and what the network
predicted (a **residual**), feed that residual back into the network, and
re-evaluate the full move set with it. The re-evaluation yields a new top-`K`,
which is simmed to make the final pick.

The intuition: each position has nuances driven by lexical facts — which words
are makeable where, with which unseen tiles — that the network cannot fully
derive, because doing so would require searching the lexicon over tiles not on
the current rack, especially for future turns. This is the network's primary
blind spot (established by the lexical-NN track; see
[lexical_features_for_value.md](lexical_features_for_value.md)). Monte Carlo
rollouts *do* consider those possibilities, honestly, via the GADDAG. When the
sims reveal that a board region is hotter or more dangerous than the network
believed, the residual encodes exactly that surprise — an indirect but
grounded clue about lexical structure. A network trained to condition on these
residuals can produce a runtime-sim-informed re-evaluation that no amount of
static input engineering can match, because the evidence comes from deep,
position-specific search.

## Where this sits

- **Roadmap Phase 4** ([roadmap.md](roadmap.md)): the one-round pipeline —
  GADDAG generates all moves, `M_pre` scores them in one cross-attention pass,
  top-`K` go to simulation. This proposal wraps that pipeline in a second
  iteration.
- **Scribblez.pdf §8.1 (Search-Derived Knowledge Buffers)**: the design doc
  envisions a buffer where truths discovered during search are recorded for the
  network to read. The residual planes are a concrete instantiation of that
  buffer, with a spatial encoding and a natural training story.
- **The belief system's iterative particle generation** (Scribblez.pdf §3.5)
  follows the same idiom: propose, gather evidence, condition on the evidence,
  re-propose. Here the proposer is `M_pre`, the evidence is sim aggregates, and
  the re-proposal is the second top-`K`.
- **Engineered lexical features**
  ([lexical_features_for_value.md](lexical_features_for_value.md)) are
  complementary, not redundant. The features attack *recall* — getting
  lexically promising moves into the first top-`K` at all. The residual loop
  attacks a different failure: the first-pass evaluation misjudged the board,
  the sims exposed it, and the re-evaluation corrects the ranking. Neither
  mechanism subsumes the other (see [Limitations](#limitations-and-caveats)).
- **`M_post`'s `OppNextPlacement` head**
  ([training_targets.h](../engine/include/scribblez/training_targets.h))
  already predicts a 15×15 mask of where the opponent's next move will place
  tiles. The heads below are that head conjoined with the game outcome — this
  proposal upgrades an auxiliary head into a load-bearing part of the decision
  loop.

## The per-square outcome heads

Two new 15×15 heads on the value models, each a per-square Bernoulli
probability:

- **Opponent danger**: for square `S`,
  `Pr[opponent's next move occupies S  AND  opponent wins the game]`.
  A high value reads as "this is a dangerous spot — to the extent the opponent
  wins more than the baseline suggests, a play here is likely the cause."
- **Self opportunity**: `Pr[our next move occupies S  AND  we win the game]` —
  hot spots for our own follow-up.

**Training targets are free from self-play logs.** For any sampled position the
log contains the next move each player made and who won, so the target for each
head is the indicator plane (placed squares of that next move) multiplied by
the win indicator. Loss is per-square BCE. An exchange or pass places no tiles,
so its plane is all zeros — the heads are per-square probabilities, not a
distribution over squares, and need not sum to anything.

Note what the conjunction mixes: a square can score high because the opponent
*plays* there often, or because playing there *wins* often. Feeding the
marginal occupancy prediction (the existing `OppNextPlacement` head) alongside
the conjunction lets the network disentangle the two.

## The residual

The Monte Carlo sim of candidate move `M` runs `S` rollouts from `M`'s
post-move state (opponent racks sampled, play continued by the rollout
policy). Each rollout observes an actual opponent reply (its placed squares)
and an outcome, so the sim yields an **empirical map** for each head — the
fraction of rollouts in which the opponent's reply occupied `S` and the
opponent went on to win — directly comparable to the network's prediction for
`M`. The residual for candidate `M` is:

```
residual_M[S] = empirical_M[S] − predicted_M[S]        (one 15×15 plane per head)
```

A large positive opponent-danger residual at `H12` means "this spot is more
dangerous than you think" — which tends to happen precisely because of lexical
possibilities the network could not consider.

### From per-candidate residuals to a re-evaluation input

The residuals are per-simmed-candidate, but the second pass must score **all
`N` legal moves** off one shared board encoding. The signal fortunately
transfers: a hot spot is mostly a lexical fact about *existing* board structure
plus the unseen pool, which is common across candidates. So the residual input
is split into a shared part and a per-move part:

- **Board-level planes** (shared): an element-wise aggregate (mean, max, or
  both) of the `K` simmed candidates' residual planes, per head. These enter
  the board encoder as extra input planes, so every one of the `N` candidates'
  cross-attention sees them. What aggregation discards — "`H12` is only
  dangerous if our move fails to block it" — the network can recover in the
  second pass, because it sees each candidate's footprint and whether it
  overlaps the hot region.
- **Per-move features** (only for the `K` simmed candidates, riding in the
  per-move embedding):
  - the candidate's own exact residual summary,
  - the **scalar value residual** — sim WLD minus predicted WLD. This is the
    most direct "the sim disagrees with you, by this much" signal; the spatial
    planes are the *where/why*, the scalar is the *how much*,
  - a simmed/not-simmed flag.
- **Confidence side-channel**: the rollout count (and per-square reply-visit
  counts if cheap). Empirical maps from a few hundred rollouts are noisy, and
  the network needs to know how much to trust a residual.

## The two-round decision procedure

1. GADDAG generates all `N` legal moves.
2. `M_pre` first pass: residual inputs zeroed (with the not-simmed flag set
   everywhere). Take the top-`K` → Monte Carlo sim each, recording the
   empirical maps and value estimates.
3. Compute residuals; build the board-level aggregate and per-move features.
4. `M_pre` second pass over **all `N` moves** with the residual inputs
   populated. Take the new top-`K`.
5. Sim any newly promoted candidates (candidates retained from round 1 reuse
   their round-1 rollouts, or top them up).
6. Final pick: best move by simulation value over the union of simmed
   candidates.

**Why the payoff is promotion, not re-scoring.** For the `K` moves already
simmed, the sims themselves provide the ranking — a re-evaluation adds little.
The value of the second pass is that it can **promote moves that were not in
the original top-`K`**: e.g. a modest-scoring play that blocks the
newly-discovered hot spot at `H12`. A one-round pipeline can never select such
a move, no matter how many rollouts it spends.

A helpful robustness property: opponent hot spots are discovered by simming
*any* candidate that fails to block them — the opponent's rollout replies land
there regardless of which of our candidates was simmed. So coverage of
opponent danger is not sensitive to the exact composition of the first top-`K`.

## Training

### `M_post` with residual inputs

The second pass needs a network trained to *use* residual inputs, which means
training rows must carry them. The semantics chosen for the residual input is
**"the evidence gathered at the decision point"**: the aggregate over the `K`
sims run from the pre-move position. This definition is uniform across all
candidates — simmed or not — which keeps the `M_pre` distillation targets
well-defined for the whole move set (a per-its-own-move residual would be
undefined for the unsimmed majority).

Data generation, per labeled position:

1. Run the round-1 pipeline at the position (top-`K` by the current model,
   `S` rollouts each).
2. Store the **empirical maps and sim value estimates** — *not* the residuals —
   alongside the `.slog` data.
3. Training targets are unchanged: final game outcome (WLD, ScoreDiff), plus
   the per-square conjunction heads from the log.

**Store empiricals, subtract fresh.** The residual depends on the current
model's own prediction, so a stored residual goes stale the moment the model
trains. The empirical maps are model-independent. At train time (and exactly
mirroring inference), the row is evaluated in two passes: a zero-residual first
pass produces the baseline prediction, the residual is formed by subtracting it
from the stored empirical map (with a stop-gradient on the subtraction), and
the second pass consumes it. Train both passes jointly with shared weights —
the zero-residual rows double as training signal that keeps the first pass from
degrading. This keeps train and inference byte-consistent, in the same spirit
as the replay-reconstruction invariant
([architecture.md](architecture.md)).

### `M_pre` with residual inputs

`M_pre`'s training story is unchanged in shape: it distills `M_post`
(roadmap Phase 4), now with the residual inputs present on both sides. For a
labeled position, the target for candidate `M` is residual-conditioned
`M_post` evaluated on `M`'s post-move state, with the same decision-point
evidence as input. The "label a subset, mask the loss" strategy from Phase 4
applies unchanged.

### The cost elephant

Residual-conditioned training rows require running `K` sims at data-generation
time. At `K ≈ 10` and a few hundred rollouts per candidate, that is thousands
of rollout-games per labeled position — even at HastyBot speeds (~ms/game),
minutes per game of self-play, a ~10³–10⁴× slowdown over plain generation.
Mitigations, all compatible:

- **Label a sparse subset of positions.** Only rows carrying residual evidence
  need the sims; the rest train with zeroed residuals (which are needed anyway
  for the first pass). The labeled fraction is a free parameter, exactly as in
  Phase 4 target generation.
- **Cheap sims for labeling.** HastyBot rollouts with modest `S`; the
  confidence side-channel tells the network the noise level, so small `S` is a
  soft degradation, not a correctness problem.
- **The generational pipeline is built for this.**
  [generational_training.md](generational_training.md) exists precisely
  because expensive generation makes per-game reuse mandatory; residual
  labeling is a (heavy) instance of the pattern, and the empirical maps are
  stored per generation like any other precomputed feature.

## Limitations and caveats

- **The sim is not unbiased ground truth.** Its "truth" is filtered through
  the opponent-rack sampling scheme (uniform over unseen tiles until the
  belief system lands) and the rollout policy (HastyBot initially). Residuals
  therefore encode a mixture of lexical blindness, rack-inference mismatch,
  rollout-policy weakness, and Monte Carlo noise. Lexical blindness is
  expected to dominate — it is the one *systematic* gap between the network
  and the GADDAG-driven rollouts — but the conflation is real and worth
  remembering when interpreting results.
- **Self-created opportunities stay out of reach.** Sim evidence covers only
  the `K` simmed post-move boards. A move whose value exists *only* because of
  structure it creates (an `S`-hook it opens), and which missed the first
  top-`K`, is never simmed — no residual can rescue it. That recall gap
  belongs to the cross-check-delta feature
  ([lexical_features_for_value.md](lexical_features_for_value.md)); the two
  mechanisms genuinely need each other.
- **Correlation with self-play rollout policy.** In HastyBot-generated
  training data, the actual opponent reply and the sim rollouts follow the
  same policy, so the empirical map is highly predictive of the logged reply.
  The conjunction-head loss will improve for that shallow reason; the metric
  that matters is **WLD / calibration improvement**, not the spatial-head
  loss.
- **Per-square SNR.** Reply footprints are ~2–7 squares of 225, so most
  squares' empirical probabilities are small and residuals there are
  variance-dominated at practical `S`. Genuine hot spots concentrate mass
  (they recur across rollouts), which is what makes them detectable; the
  confidence inputs exist so the network can discount the rest.
- **One extra round, not a fixed point.** The loop could iterate (re-sim,
  re-residual, re-evaluate), but each round costs a sim budget and the
  evidence gain shrinks. Two rounds is the design point; more only if
  measurement demands it.

## De-risking: the kill-test

The load-bearing hypothesis is narrow: *conditioning on sim-derived residuals
improves the value model's outcome prediction.* That is testable offline,
cheaply, before any agent or `M_pre` work:

1. Take a modest set of self-play positions; for each, generate residual
   labels (top-`K` by the current `M_post` over candidate post-move states,
   HastyBot rollouts, empirical maps).
2. Train `M_post`-with-residual-inputs vs. the plain baseline on identical
   data.
3. Compare held-out WLD loss and calibration (the Phase 3 machinery).

If the residual-conditioned model shows no WLD improvement, the whole loop is
moot — and that is learned without touching `M_pre`, the two-round agent, or
any selection-time plumbing.
[monte_carlo_sim_tool](../engine/apps/monte_carlo_sim_tool.cpp) (the
Positions-tab ground-truth generator, see
[react_dashboard.md](react_dashboard.md)) already contains most of the rollout
machinery this experiment needs.

## Implementation roadmap

| Step | Build | Depends on |
|------|-------|-----------|
| 1 | Conjunction heads on `M_post` (targets from logs; per-square BCE). Independent value as probes even if the loop is never built. | — |
| 2 | Sim machinery emits per-square empirical maps + value estimates (extend the `monte_carlo_sim_tool` rollout core into a reusable `SimRunner`); storage format for empirical maps alongside `.slog`. | 1 |
| 3 | **Kill-test** (above): offline residual-conditioned `M_post` vs. baseline. **Go/no-go gate for everything below.** | 2, Phase 3 eval machinery |
| 4 | Input-encoder extension (residual planes, scalars, confidence, flags); two-pass training loop (zero-residual first pass, stop-gradient subtraction); residual labeling integrated into generational data generation at a sparse position fraction. | 3 |
| 5 | `M_pre` inherits the heads and residual inputs; distillation from residual-conditioned `M_post`. | 4, roadmap Phase 4 |
| 6 | Two-round agent (the decision procedure above, with round-1 sim reuse); match-play eval vs. the one-round agent (Phase 3 agent-eval harness). | 5 |

Steps 1–3 are cheap relative to what they de-risk and are worth doing early;
steps 4–6 ride the Phase 4 timeline.

## Open questions

- **Aggregation function** for the board-level planes: mean, max, or both
  stacked. Max preserves "some candidate exposed this danger"; mean preserves
  magnitude. Cheap to ablate in the kill-test.
- **Budget split**: `K` and `S` per round, and whether round 2 should sim a
  smaller promoted set. Round-1 sim reuse makes the marginal round-2 cost
  proportional to the number of *newly promoted* candidates only.
- **Whether the scalar value residual alone captures most of the win.** If the
  per-move scalar residuals (no spatial planes) already deliver the WLD
  improvement, the spatial machinery can be deferred — the cheap-before-rich
  sequencing of [lexical_features_for_value.md](lexical_features_for_value.md)
  applies here too.
