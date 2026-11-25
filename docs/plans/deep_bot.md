# DeepBot: continuous training on searched labels

**Status: proposed 2026-11-24, not built.** What each phase measures shapes
the next one. No pass or kill thresholds are fixed in advance.

**Decision.** Replace deep_bot_generation's recipe with an AlphaZero-like one,
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
- **Generators play the newest release**, as SubtreeBot's generations workload's did. The
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
- **The leaf never sees what it ranks.** DeepBot uses the leaf to rank a
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

| Input | Known mode | Belief mode |
|---|---|---|
| The opponent's per-tile counts (27) | the leave's counts | their expected counts under the belief |
| The opponent's static leave value: mean and SD (2) | the leave's value; SD 0 | the exact mean and SD under the belief |
| The belief's 29 parameters: the 27 per-tile tilts, `a` and `b` (`engine/include/belief/leave_params.h`) | zeros | the leave inference model's output |
| The **unseen tiles from my side**: every tile not on the board or my leave. Their counts replace the pool's. | set | set |
| The opponent's leave length, and a **mode flag** | set | set |

**The count slots are shared, so their meaning must be too.** A concrete
leave is its own expectation, so expected counts mean the same thing in both
modes. The tilts would not work in those slots. They are relative to the
pool's prior, so a 0 there would mean "as likely as a uniform draw" in belief
mode but "absent" in known mode. So the tilts get slots of their own, zeroed
in known mode.

**The belief's summaries cost nothing extra.** `LeaveSampler`
(`engine/include/belief/leave_sampler.h`) makes a pass over every leave to
sample from the belief, and DeepBot makes that pass every turn already. Its
`LeaveMoments` already returns the leave value's exact mean and SD. Expected
counts come from the same pass with a small extension. The belief is the same
for every candidate at a turn, so the pass runs once per turn.

**The join planes are ablated** in Phase 2: format version 3 is trained with
them and without them, and they go if they don't earn their place. The
encoder header makes the case for them as a conv trunk's weakness, joining a
square's cross-check letters with who holds them. A transformer can make that
join from the cross-checks and the counts. Release1's transformer leaf trained
with them, and they have never been ablated on a transformer.

If the ablation keeps them, only mine and the unseen tiles' stay. In belief
mode the opponent's join planes would have to become the probability that the
leave holds a fitting tile, which means inclusion–exclusion over the belief at
every hook square.

**Training rows.**
- **Belief-mode rows** take a target averaged over the belief. A searched
  candidate's mean over its sampled opponent racks is exactly that.
- **Known-mode rows** need a target conditioned on the actual leave. A single
  sample of the search supplies it: sample *i* is a world in which the
  opponent holds rack *i*, and its value at the cutoff is a target for that
  world. That target is noisy, but it costs nothing extra.

The trainer draws the two modes half and half. The belief comes from a
`.belief` sidecar beside each `.slog` (decided 2026-11-25), since its summaries
cost a pass over every leave, too much to repeat per row per epoch. For
HastyBot's games the trainer writes it under the tag's pinned leave model. For
DeepBot's own games DeepBot records the belief it acted on, so a row carries
its own release's belief rather than the newest. The details are in
[deep_bot_belief_leaf.md](deep_bot_belief_leaf.md#1-the-belief-sidecar).

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
SubtreeBot's generations workload did a version of this for its policy data
(`pool_positions_per_game`).

**A greedy turn** values every legal move under the belief-input leaf and
plays the best. That costs about as many evaluations as there are legal
moves:
- the known positions' median is about 550;
- a rack with a blank has thousands.

That is around three times today's 160 a turn, and it is affordable (see the
cost below).

**Where a prefilter applies:**
- **The game's greedy turns:** no prefilter.
- **A labeled turn's root:** no prefilter. Every legal move gets the leaf's
  single evaluation.

  These two places are where the prefilter research gets its data.
- **Inside the search's plies:** static equity's top `K`, with `K` of 50 to
  100, as a stopgap until the research replaces it. Without it, a row would
  cost about ten times as much (see the cost below). The search's
  continuation then differs from the game's policy only where static equity's
  top `K` misses the leaf's best, and the counts below measure how often that
  happens.

### The prefilter as research

**The goal** is a set of fast, independent decision stumps S1, S2, …, Sn
over different inputs, so that their misses are largely unrelated. Each stump
picks a subset of the legal moves, and the union of those subsets should
almost always hold the leaf's best. Static equity's top `K` is S1.

**The stumps are non-neural.** A stump runs on every legal move of every turn,
so it must cost far less than the leaf evaluations it saves. A network, even a
small one, costs too much to be worth it there.

**What makes it measurable:** the union's recall is S1's recall plus
P(S1 misses) × P(S2 finds the best | S1 misses), and so on for each further
stump. A stump's cost, the size of the subset it picks, can be measured on any
positions.

**What the workload records** for that research, which runs separately over
the data:
- **Counts, at every greedy turn and every labeled turn's root**, since both
  value every legal move anyway:
  - A, the static equity rank of the leaf's best;
  - B, the number of tiles in the bag;
  - the count of turns at each (A, B);
  - at each of a few `K` (5, 10, 20, 50, 100), how far the leaf's best
    within static equity's top `K` falls short of the overall best.

  The shortfall is the cost of a miss: missing the best by 0.001 of win
  probability doesn't matter, so rank alone overstates the misses. Choosing
  `K` trades the shortfall against evaluations.
- **The misses:** every turn whose best is outside static equity's top
  `K_keep`, a parameter defaulting to 10.
  - A miss at any larger `K` is a subset of these.
  - Each is kept as its game's record, the turn index and the release that
    played it, since inputs are rebuilt by replay. A later leaf or stump can
    then be scored on them.
  - They go in a directory under the tag that the window's eviction never
    touches.
  - They are the conditional set the recall identity needs.

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
4. **Race.** The search runs in rounds and drops candidates once they are
   clearly behind. The policy-set pool labeler did this; it was deleted, and
   sim/pool_labeler.h is in git history. A dropped candidate keeps a rough
   value that leans low, because it was dropped for looking bad.

**What a labeled turn yields:**
- the move played (the search's best, which is where the policy improves);
- a belief-mode row per valued candidate;
- known-mode rows from the samples.

The sibling rows teach the ranking that DeepBot uses the leaf for.

**Cost.** A row costs about `R × (k × K + 1)` leaf evaluations, where `R` is
the average number of draws per candidate after racing and `K` is the plies'
prefilter. A game's greedy turns value every legal move, about 22 × 500.

| Setting | Per row | Per game of greedy turns |
|---|---|---|
| `K` = 50, `R` ≈ 6, `k` = 2 | ~600 | ~11K |
| `K` = 50, `R` ≈ 6, `k` = 4 | ~1,200 | ~11K |
| no prefilter in the plies, `R` ≈ 6, `k` = 2 | ~6K | ~11K |

**How many candidates enter the search** is a knob. Every root move already
has the leaf's single evaluation, so the race can start from the leaf's
ranking and take its best few. That is not a static-equity prefilter, so it
does not bias the counts above.
- Searching 50 candidates costs about 30K evaluations at `k` = 2. At one
  labeled turn per game, the greedy turns are then about a quarter of the
  cost.
- Racing all ~500 root moves would cost about ten times as much per search,
  cutting a generation to a few hundred labeled positions.

**Fewer labeled turns per game buys diversity.** A search's sibling rows are
already correlated, and more games is what adds new positions. With the greedy
turns at about a quarter of the cost, going below one labeled turn per game
starts to cost real budget. `p` near 1/22, about one labeled turn per game, is
the starting point.

**Counting the games, a row costs about four to eight times today's.** At
today's budget of about 70M evaluations, `k` = 2 with 50 candidates a search
gives roughly 1,700 games and 85K rows a generation. What it buys:
- **Lower variance.** The targets are averaged values instead of single 0/1
  game results. How much lower depends on the spread of the cutoff value from
  one draw to the next, which Phase 0 measures. If that spread were 0.15, six
  draws would leave a variance around 0.004, against about 0.25 for a game
  result.
- **Siblings.** Each search yields rows for the moves that were not played.

`p`, `R`, `K` and the number of candidates per search trade rows against
positions within any budget.

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

Generators take the tag's newest release, as SubtreeBot's generations
workload did (subtree_bot_generations.md; its code is in git history as
workloads/deep_bot.py), rather than release 0
for the tag's life. The
trainer's window spans the recent generations whatever release produced
them, and each row records its release.

This is a new workload, `deep_bot`: deep_bot_generation's definition is a
fixed policy.

## The Positions tab

**The opponent's rack panel** shows generation `N` under release `N`'s leave
inference model, because that model is what played generation `N`'s games.
The charts go in `dashboard.db` keyed by (generation, position).
- scz#544 charts the fixed release 0 model. That is right for
  deep_bot_generation, whose policy is fixed, and it can merge as is.
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

**Cost of the column.** Without THAN, the known positions hold about 14.2K
legal moves. With 16 first-round draws, `K` = 50 and `k` = 2, that is about
23M evaluations, around 6 minutes of one GPU per analyzed generation; `k` = 4
is about twice that. Racing adds a little for the contenders.

**Unchanged:**
- deep_bot_leaf_eval keeps its hasty rollouts, which are its leaf's exact
  target: that leaf trains on HastyBot's self-play.
- Moving its `hasty_rollouts.json` from the repo into the tag is optional
  cleanup.

**THAN** leaves the known positions. Its ~20,000 legal moves would be more
than half of every rollout bill, and the set exists to support the tab, not to
represent any distribution of positions.

## Phases

0. **Measurements**, on the version 3 leaf of phase 2's runs (release1 was
   deleted). Measure:
   - the spread of the cutoff value from one draw to the next, at `k` = 2 and
     4;
   - the distribution of legal-move counts;
   - leaf throughput at the batch sizes the greedy turns make.
1. **Small changes now:** remove THAN (scz#546); merge scz#544 (done).
2. **The belief-input leaf**, sliced in
   [deep_bot_belief_leaf.md](deep_bot_belief_leaf.md):
   - the format version 3 encoder, with the belief's summaries from
     `LeaveSampler`'s pass;
   - the trainer's two modes on game-result targets;
   - the join plane ablation, with and without;
   - belief-mode loss against DeepBot's 16-draw average on held-out
     positions.
3. **The greedy player:** DeepBot in belief mode, one evaluation per
   candidate. Measure its speed and its strength against today's DeepBot.
4. **The labeled-turn search** in the engine: paired draws, racing, `k`
   greedy plies, a known-mode cutoff. It emits belief-mode and known-mode
   rows.
5. **The new workload:**
   - generators follow releases and label turns with probability `p`;
   - the trainer consumes the searched rows, with the game result mixed in;
   - the prefilter counts and the kept misses.
6. **The Positions tab:**
   - the opponent's rack per release;
   - the search columns from the measure role.

## Open questions

- **`k`, `R`, `K`, `p`,** and the number of candidates a search values
  against the budget. Phase 0's spread and prefilter measurements set the
  first values.
- **The prefilter stumps** beyond S1, all non-neural, and the `K` for S1
  inside the search's plies, chosen from the counts and the kept misses.
- **The weight of the game result** in the played candidate's target.
- **The leaf's architecture:** whether the speed of a conv leaf outweighs
  the transformer's quality once evaluations are the whole cost.
