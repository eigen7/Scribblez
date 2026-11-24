# LeafBot: continuous training on searched labels

**Status: proposed 2026-11-24, not built.** What each phase measures shapes
the next one. No pass or kill thresholds are fixed in advance.

**Decision.** Replace leaf_bot_generation's recipe with an AlphaZero-like one,
in four parts:

- **The leaf takes the opponent's leave as a belief**, the leave inference
  model's output, as well as a concrete leave. Half its training rows hide the
  concrete leave, so one network serves both uses.
- **Self-play plays a cheap greedy move** (the best move under the belief-input
  leaf) at most turns. With probability `p` a turn is instead a **labeled
  turn**: a search values the turn's candidates, the search's best is played,
  and every valued candidate becomes a training row.
- **A search plays each candidate a few plies on with the greedy policy**,
  over sampled opponent racks, and stops at the leaf's value. There are no
  HastyBot plies anywhere.
- **Generators play the newest release**, as deep_bot's already do. The
  training window spans many releases.

The Positions tab follows. It drops the hasty rollouts for the same search
run at the known positions, and it shows each generation under that
generation's own models.

## Why

- **Self-play is slow.** Every self-play turn values static equity's top 10
  candidates over 16 drawn opponent leaves each, so it costs 160 leaf
  evaluations. The leaf in release1 is the width-64 transformer, about 62K
  evaluations a second on the laptop GPU (TensorRT FP16, batch 1K). A
  generation is 20,000 games of about 22 turns that use the leaf, so roughly
  70M evaluations, or 20 minutes of one GPU if it ran at that rate.
- **Most of that buys noise.** Each row's target is its game's result, a 0
  or a 1. Every turn of a game shares it. A generation yields about 400K such
  rows, about 175 evaluations each.
- **The prefilter hides the leaf's answers.** In release1's generation 70
  analysis, the leaf's best move is outside static equity's top 10 in 7 of
  the 19 known positions: 75th at Q(I), 66th at DINER. The positions were
  picked as hard cases, so this overstates ordinary play. Still, today's
  self-play can never play those moves.
- **The leaf never sees what it ranks.** LeafBot uses the leaf to rank a
  turn's candidates. It trains only on positions after moves that were
  actually played, which are good moves.
- **The Positions tab's rollout columns measure a different target.**
  They are HastyBot played out to the game's end, not what the leaf trains on,
  so the table cannot say whether the leaf learned its training data.

## The belief-input leaf

The leaf's input (engine/include/encoding/leaf_eval_input_encoder.h, format
version 2) assumes both leaves are known. When Bob values his candidates,
the hidden leave is Alice's. His belief about it is the same for every
candidate, so a ply costs one leave inference call and one leaf evaluation
per candidate, instead of 16 per candidate.

**Which inputs depend on the opponent's leave:**
- their per-tile counts (27);
- their static leave value (1);
- their join planes (4);
- the **pool**, defined today as the tiles on neither the board nor either
  leave: its counts (27) and its join planes (4).

The pool size and the bag-after-refills buckets depend only on the leave's
length, which a player knows, so they stay.

**The input at format version 3:**

| Input | When it is set |
|---|---|
| The **unseen tiles from my side**: every tile not on the board or my leave. Their counts and join planes replace the pool's. | Both modes |
| The **belief**: the leave inference head's 29 parameters, and the opponent's leave length. | Both modes |
| The opponent's concrete counts, static leave value and join planes. | Known mode only; zeros otherwise |
| A **mode flag**. | Both modes |

The new unseen-tile inputs read the same whether the opponent's leave is
known or not. The concrete inputs carry what knowing the leave adds.

**Training rows.**
- **Belief-mode rows** take a target averaged over the belief. A searched
  candidate's mean over its sampled opponent racks is exactly that.
- **Known-mode rows** need a target conditioned on the actual leave. A single
  sample of the search supplies it: sample *i* is a world in which the
  opponent holds rack *i*, and its value at the cutoff is a target for that
  world. That target is noisy, but it costs nothing extra.

The trainer draws the two modes half and half. It computes the belief by
replaying the game under its current leave inference model. That is the
model the leaf is served with, and the trainer trains both models anyway.

**Before any search exists,** the encoder change can be trained on
today's game-result targets, which suit both modes. That checks that the
dual-mode leaf works before anything depends on it.

## Self-play

**Today** self-play keeps every turn of every game. The trainer samples 5
turns per game per generation over a 4-generation window, so each turn trains
about once.

**Proposed:** each turn is a labeled turn with probability `p`. This is
KataGo's playout cap randomization, applied here: search on a random fraction
of turns, which alone make training rows, and cheap play everywhere else.
deep_bot already does a version of this for its policy data
(`pool_positions_per_game`).

**A greedy turn** values a prefilter of `K` candidates under the belief-input
leaf and plays the best. Valuing every legal move is the cleanest choice,
but it costs about as many evaluations as there are legal moves:
- the known positions' median is about 550;
- a rack with a blank has thousands.

That is around three times today's 160 a turn. So the greedy turns keep a
prefilter:
- static equity's top `K`, with `K` of 30 to 50;
- or the move proposal model (docs/plans/sim_residual_feedback.md).

Phase 0 measures how often each misses the all-moves best on ordinary
self-play positions.

**A labeled turn's search** works as follows:
1. **Draw racks.** For each candidate, draw `R` opponent racks from the
   mover's belief. The draws are paired across the candidates: draw *i* is the
   same rack and bag order for every candidate. This is the pairing that
   engine/include/sim/hasty_rollout_task.h's batches already use.
2. **Play the plies.** Each draw plays `k` plies of greedy turns. Each player
   values candidates with the leaf in belief mode, because a player who sees
   the other's actual rack would not be playing the self-play policy.
3. **Stop at the cutoff.** After the `k` plies, the simulator knows both
   leaves, so it values the position with the leaf in known mode. A game that
   ends within the plies takes its exact result. Turns with the bag empty
   still go to the endgame solver, as they do today.
4. **Race.** The search runs in rounds, as `PoolLabeler` does, and drops
   candidates once they are clearly behind. A dropped candidate keeps a rough
   value that leans low, because it was dropped for looking bad.

**What a labeled turn yields:**
- the move played (the search's best, which is where the policy improves);
- a belief-mode row per valued candidate;
- known-mode rows from the samples.

The sibling rows teach the ranking that LeafBot uses the leaf for.

**Cost.** A row costs about `R × (k × K + 1)` leaf evaluations, where `R` is
the average number of draws per candidate after racing. A game's greedy turns
cost about `22 × K`.

| Setting | Per row | Per game of greedy turns |
|---|---|---|
| `K` = 50, `R` ≈ 6, `k` = 2 | ~600 | ~1.1K |
| `K` = 50, `R` ≈ 6, `k` = 4 | ~1,200 | ~1.1K |

A search over 50 candidates costs about 30K evaluations at `k` = 2, so at one
labeled turn per game the greedy turns are about 4% of the cost. **Labeling
fewer turns per game is therefore nearly free, and it buys diversity**: a
search's 50 sibling rows are already correlated, and more games is what adds
new positions. `p` near 1/22 gives about one labeled turn per game. Lower is
reasonable.

**Per row, this is three to seven times today's cost.** What it buys:
- **Lower variance.** The targets are averaged values instead of single 0/1
  game results. How much lower depends on the spread of the cutoff value from
  one draw to the next, which Phase 0 measures. If that spread were 0.15, six
  draws would leave a variance around 0.004, against about 0.25 for a game
  result.
- **Siblings.** Each search yields rows for the moves that were not played.

At today's budget of about 70M evaluations, `k` = 2 gives roughly 2,300 games
and 115K rows a generation. `p`, `R`, `K` and the number of candidates per
search trade rows against positions within any budget.

**Bias.** A target that stops at the leaf's own value inherits the leaf's
errors where it stopped. AlphaZero avoids this by training value on game
results. Here three things anchor the targets:
- games that end within the plies take exact results;
- the endgame solver's turns are exact;
- the labeled turn's game is played out to the end anyway, so its result can
  be mixed into the played candidate's target.

The mixing weight is a knob to measure, not a fixed choice.

**Other levers on speed:**
- **The leaf's size.** A 32×4 conv leaf runs at about 540K evaluations a
  second against the transformer's 62K. deep_bot_leaf_eval's 2×2 comparison
  found the arms tied on loss.
- **Where encoding runs.** Once a turn is one evaluation per candidate, the
  CPU encoder (about 71K rows a second per thread) needs several threads to
  keep a faster leaf fed. The GPU encoder from the GPU hasty rollouts (about
  3M rows a second) is the alternative.

## Continuous training

Generators take the tag's newest release, as deep_bot's generators do
(workloads/deep_bot.py), rather than release 0 for the tag's life. The
trainer's window spans the recent generations whatever release produced
them, and each row records its release.

This is a new workload. leaf_bot_generation's definition is a fixed policy,
and the live release1 tag runs under its stored params.

## The Positions tab

**The opponent's rack panel** shows generation `N` under release `N`'s leave
inference model, because that model is what played generation `N`'s games.
The charts go in `dashboard.db` keyed by (generation, position).
- scz#544 charts the fixed release 0 model. That is right for
  leaf_bot_generation, whose policy is fixed, and it can merge as is.
- The new workload's panel keys by release.

**The value table's rollout columns** become the labeled-turn search run at
the known positions under release `N`. It is the same engine code with more
draws, and it answers whether the leaf learned what its training data says.
- The measure role computes it with each analysis. That role already has any
  number of slots and already delivers files into the tag.
- The controller indexes the delivered files into `dashboard.db`.
- The columns are omitted where the search has not run.

This replaces the generalized, non-singleton rollouts workload we discussed
before: in a continuous run the column changes with every release, so it
belongs with the per-release analysis.

**Cost of the column.** Without THAN, the known positions hold about 8.8K
legal moves. With 16 first-round draws, `K` = 50 and `k` = 2, that is about
14M evaluations, around 4 minutes of one GPU per analyzed generation; `k` = 4
is about twice that. Racing adds a little for the contenders.

**Unchanged:**
- deep_bot_leaf_eval keeps its hasty rollouts, which are its leaf's exact
  target: that leaf trains on HastyBot's self-play.
- Moving its `hasty_rollouts.json` from the repo into the tag is optional
  cleanup.

**THAN** leaves the known positions. Its ~20,000 legal moves would be two
thirds of every rollout bill, and the set exists to support the tab, not to
represent any distribution of positions.

## Phases

0. **Measurements**, on release1's models. Leaves are drawn where the belief
   input does not exist yet. Measure:
   - the spread of the cutoff value from one draw to the next, at `k` = 2 and
     4;
   - how often static equity's top `K` (and the move proposal model)
     misses the all-moves best on self-play positions;
   - the distribution of legal-move counts;
   - leaf throughput at the small batches a greedy turn makes.
1. **Small changes now:** remove THAN; merge scz#544.
2. **The belief-input leaf:**
   - the format version 3 encoder;
   - the trainer's two modes on game-result targets;
   - belief-mode loss against LeafBot's 16-draw average on held-out
     positions.
3. **The greedy player:** LeafBot in belief mode, one evaluation per
   candidate. Measure its speed and its strength against today's LeafBot.
4. **The labeled-turn search** in the engine: paired draws, racing, `k`
   greedy plies, a known-mode cutoff. It emits belief-mode and known-mode
   rows.
5. **The new workload:**
   - generators follow releases and label turns with probability `p`;
   - the trainer consumes the searched rows, with the game result mixed in.
6. **The Positions tab:**
   - the opponent's rack per release;
   - the search columns from the measure role.

## Open questions

- **`k`, `R`, `K`, `p`,** and the number of candidates a search values
  against the budget. Phase 0's spread and prefilter measurements set the
  first values.
- **The greedy turns' prefilter:** static equity's top `K`, the move
  proposal model, or every legal move if a faster leaf makes that affordable.
- **The weight of the game result** in the played candidate's target.
- **The leaf's architecture:** whether the speed of a conv leaf outweighs
  the transformer's quality once evaluations are the whole cost.
