# DeepBot: two-level simulation with a policy-set model

**Status: proposed 2026-10-14; not built, and not yet placed in the
[roadmap](../roadmap.md).** DeepBot plays standard Scrabble, with hidden
leaves. This document is a design to guide development. Its measurements
inform the next step; no pass or kill thresholds are fixed in advance.

Write **Alice** for the player to move and **Bob** for the opponent.

## The turn

1. **Candidates.** The [policy-set model](#the-policy-set-model)
   proposes `N_A` candidates for Alice from all her legal moves.
2. **Bob's racks.** Alice samples `R_A` racks for Bob (about 100) from her
   [belief](#beliefs). They are fixed for the turn and shared by every
   candidate.
3. **Bob's move.** For each candidate `a` and each Bob rack `r`, Alice
   predicts Bob's reply `β(a, r)` by building [Bob's
   subtree](#bobs-subtree), which models Bob's decision on Bob's
   information.
4. **Alice's sims.** For each `(a, r)`, Alice runs several
   [sims](#sims), each with different replenishment draws:
   1. Bob plays `β(a, r)`.
   2. Alice and Bob each play one more move by a mechanical rule.
   3. The leaf is evaluated.

   Her value `q(a)` is the mean over racks and sims.
5. **The pick.** Alice plays the candidate with the highest `q(a)`.

## Bob's subtree

Bob's subtree answers one question: given that Alice played `a`, and Bob
holds `r`, what does Bob play?

1. **Alice's racks, from Bob's side.** Bob samples `R_B` racks for Alice
   (about 20) from *his* belief. That belief conditions on the public history
   *including `a`*, and its pool is the tiles Bob cannot see, given `r`.
2. **Bob's candidates.** The policy-set model, evaluated from Bob's
   information set (`r` plus the public state), proposes `N_B` moves.
3. **Bob's sims.** For each candidate `b` and each of Alice's sampled racks:
   1. Bob plays `b`.
   2. Alice and Bob each play one more move by the mechanical rule.
   3. The leaf is evaluated from Bob's side.

   All of Bob's candidates use the same `R_B` racks and draws, so they are
   compared on paired samples.
4. **The choice.** `score(b)` is the mean over Alice's sampled racks, and
   `β(a, r) = argmax_b score(b)`.

**Only the move crosses back.** Alice's sims use `β(a, r)`, never Bob's
scores. Bob's scores are expectations over Bob's belief about Alice's rack,
not over Alice's actual rack, so they belong to Bob's decision and nowhere
else.

![Two-level simulation: Alice's evaluation spawns a subtree for Bob at each sampled rack, and only Bob's chosen move crosses back](../images/deep_bot_many_tree.svg)

*Each triangle is one player's evaluation, colored by whose beliefs it runs
on: red for Alice's, blue for Bob's. Each node is colored by whose turn it is,
and labeled with Alice's tiles, then Bob's, with `?` for tiles unknown to the
evaluation's owner.*

*Alice starts at `x ?`. She holds `x` and cannot see Bob's rack. The
policy-set model proposes `a` among her candidates. Alice samples Bob's rack
from her own belief, `y` or `z`. At `x z` it is Bob's turn, and Alice cannot
choose Bob's move with her own values, so she **spawns** Bob's subtree, rooted
at `? z`: Bob holds `z` and does not know Alice's tiles.*

*In that subtree, the policy-set model proposes Bob's set, `b₁`, `b₂`, `b₃`
and others. Bob samples racks for Alice from his own belief, which reads what
`a` reveals (`x̂₁`, `x̂₂`, …). For each of his candidates, he sims: Alice
plays hasty, Bob plays hasty, and the win table values the leaf. Each
candidate scores the mean over Alice's sampled racks, and Bob's move is the
argmax, here `b₁`.*

*Only that move crosses back. In Alice's evaluation Bob plays `b₁`; Alice
samples replenishments, both players continue with hasty moves, and the leaf
values average into `q(a)`. Every sampled rack gets its own subtree for Bob
(`y` at right).*

## Sims

**The mechanical rule** is HastyBot's: static equity over the mover's own
legal moves. It reads only the mover's rack, so the moves it makes cannot
depend on the opponent's hidden tiles. In the whole design, only two
decisions search: Alice's at the root and Bob's in his subtree. Each is made
on its own player's information.

**The leaf: a win table.** After a sim's fixed moves, a fast function turns
the position into a win probability, as Macondo's simmer does:

`P(win) = table(spread + leave value, tiles unseen)`

- **The spread** is from the side to move, plus the static leave value of
  that side's remaining rack, as Macondo adds its "leftover".
- **Tiles unseen** are the bag plus the opponent's rack, from the side to
  move.
- **If the game ends** inside the sim, the outcome is exact.

The probability is then taken from the side that owns the evaluation (Bob in
his subtree, Alice in hers), as `1 − P` when the other side is to move. The
leaf never reads the opponent's hidden tiles beyond their count, so it
fits a game in which neither player sees the other's leave. It is a table
lookup, so a sim's cost is its mechanical moves.

**The win table is versioned.** It is fitted from `.slog` outcomes of the
current policy: `P(win | spread + leave value, unseen)`, bucketed, smoothed and
made monotone in spread, in the CSV format the engine already reads
([sim/win_pct_table.h](../../engine/include/sim/win_pct_table.h)). The win
probability at a given spread shifts as play gets stronger, so each policy
generation gets a new version, and everything it values records the version
it used. A trained network that reads the board may replace the table later.

**Pairing.** Samples come from variates keyed by `(sample index, draw
ordinal)`:

- **Bob's racks** come from Alice's unseen pool, which does not depend on her
  candidate. So every candidate meets exactly the same `R_A` racks, and
  comparisons between candidates are paired.
- **Within Bob's subtree,** all of his candidates share the same `R_B` racks
  for Alice and the same draws.
- **Replenishments** map a variate to an index in a canonical sorted pool.
  Candidates whose pools coincide share draws exactly; where the pools
  differ (because different tiles were played) the pairing is approximate.

A seeded, indexable draw sequence replaces `Bag::draw()` in sim code.

## The policy-set model

**Interface.** Given a position from the chooser's information set, all legal
moves and a size `N`, return a set `S` of `N` moves.

**Objective.** Choose `S` to maximize the expected best value in the set,
`E[max_{a∈S} v(a)]`, where `v` is the move's value under the chooser's
uncertainty about it. This differs from taking the `N` moves most likely to
be best one at a time: a set of near-identical moves adds little over its
best member, so the objective rewards diversity. Of a strong play and four
near-copies of it, the set is better with one copy replaced by a different
kind of move, such as a setup that wins in other worlds.

Two forms of the model meet this objective. Both build on the move set
model's architecture (the board trunk, the move encoder and the cross-check
encoder), and both train on the same labeled pools ([Labels](#labels)).
Which works better is measured.

### Form 1: greedy selection, with cached steps

The expected max of a set of win probabilities is monotone and submodular, so
greedy construction comes within a factor of `1 − 1/e` of the best set. The
policy-set model picks one move at a time, each maximizing its predicted gain
over the moves already chosen:

`gain(a | S) = E[max(0, v(a) − max_{s∈S} v(s))]`

At `S = ∅` this reduces to ranking by value. A stage conditions each move's
query on the tokens of the moves already chosen. The evidence-fusion stage
built for [sim_residual_feedback.md](sim_residual_feedback.md) already
conditions on a set of move tokens. There the tokens carried sim results;
here they are chosen moves alone. Its proves-best head is the same quantity
with "simmed" in place of "chosen".

Built naively, `S` takes `N` sequential network calls. Most of each call does
not depend on `S`: the board trunk, the move encodings and the cross-check
encodings are computed once per position and cached. Each step reruns only
the set-conditioning stage and the gain head, a small fraction of a full
pass. This is the cache-graph and step-graph split the evidence path already
uses ([evidence_staging.h](../../engine/include/agent/evidence_staging.h)).
Total compute is about one full pass plus `N` light ones, but the steps
remain serial.

### Form 2: a one-shot slot decoder

The network outputs the whole set in one pass, in the manner of set
prediction (DETR; Carion et al., 2020):

- `N_max` learned query slots cross-attend to the encoded moves and
  self-attend to each other.
- Each slot points to one move, through a softmax over the legal moves of the
  dot product between the slot's query and each move's embedding.
- Because the slots attend to each other within the pass, training teaches
  them to divide the moves among themselves. Diversity comes from joint
  training, not from sequential conditioning.

**The loss is coverage of the best move.** Let `q(a)` be the label
distribution over which move is best, and `p_k(a)` the probability that slot
`k` points to `a`. With the first `N` slots,

`L_N = −Σ_a q(a) · log(1 − Π_{k ≤ N} (1 − p_k(a)))`

which is the negative log-probability that the set contains the best move.
Once one slot covers a move, another slot pointing there adds almost nothing,
so across positions that look alike but have different best moves, the slots
spread over the plausible winners. This is the mechanism of multiple-choice
learning: several outputs trained so that at least one is right. The loss
counts a set whose best move falls just short of the true best as a miss;
soft labels soften that ([Labels](#labels)), and expected regret,
`v(a*) − max_{a∈S} v(a)` with softmax pointers, is the alternative if margins
need more weight.

**Any `N` from one pass.** Training sums the loss over prefixes,
`Σ_N w_N · L_N`. Slot 1 then learns the best single move, slots 1 and 2 the
best pair, and so on, so the first `N` slots serve any `N ≤ N_max`: Alice's
`N_A` and Bob's `N_B` come from the same pass. At inference, a slot whose
move an earlier slot already took falls back to its next choice.

### Many sets at once

DeepBot builds many sets per turn. Bob's subtrees alone need one per
`(a, r)`, about 100 per candidate. With either form, each network call is one
batched call over all the sets in progress. Form 1 makes `N` such calls, all
but the first of them light; form 2 makes one.

### Labels

A labeled position is a pool of candidate moves with values from sims, run
from the chooser's information set: the opponent's leave is drawn from the
chooser's [leave inference](#the-leave-inference-model) belief, with a
uniform refill, and the sims are paired across moves.

- **Candidates: every legal play and every distinct exchange, plus the pass,
  including blank plays.** A blank multiplies the moves through its
  designations, so one prune applies: a blank move (a play that places the
  blank, or an exchange that returns it) is dropped when its score is 40 or
  more points below the greedy move's, the highest-scoring play. An exchange
  scores 0 under this rule.
- **Pools need the tail.** The pool must include moves the prior ranks low
  (exchanges, short plays, setups), or the policy-set model never learns that
  they can win. The exhaustive survey recipe of
  [sim_labeled_candidates.md](sim_labeled_candidates.md) (every legal play,
  screened with racing) produces this shape of data.
- **The labels can see moves the rollout policy never plays.** Every
  candidate is simmed directly, rather than reached by a rollout policy that
  would have to choose it. A setup that static equity ranks low is labeled by
  its own outcome, provided its payoff arrives within the sim's horizon.
- **Form 1's targets: subset assembly.** Any subset `S` of the pool, plus a
  held-out move `a`, gives a row with target `max(0, v(a) − max_S v)`,
  computed from the paired values. One pool yields many rows.
- **Form 2's targets: a soft best move.** `q(a)` is the probability that `a`
  is best under the paired sim statistics: by bootstrapping over the paired
  samples, or by a normal approximation to the paired differences against
  the leader. Near-equal moves share the mass, so the slots learn that either
  will do. Racing stops weak moves early, on fewer rollouts, and both
  estimates account for that, since a move far below the leader rarely comes
  out on top.

**Label rollouts stop at the win table.** Played to the end of the game, a
rollout is slow and its outcome noisy. A label rollout instead plays `K`
plies by the mechanical rule and ends at the [win table](#sims), whose version
the pool records.

**Generations.** Once DeepBot runs, its own evaluations label positions, and
each policy-set generation trains on the previous agent's labels, as the move
proposal model's generations do.

## Beliefs

Two beliefs feed the design: Alice's over Bob's rack (step 2 of the turn), and
Bob's over Alice's rack given `a` (step 1 of his subtree). Both come from one
model.

### The leave inference model

The **leave inference model** predicts the opponent's **leave**, the
tiles they kept from their last move, from the observer's view of the game.
The opponent's rack is that leave plus a refill drawn uniformly from the rest
of the observer's unseen pool, which is exact: the refill is a uniform draw.
The leave's length is public: the opponent's rack size minus the tiles their
last move played or exchanged (a pass keeps the whole rack).

**Inputs.** The leave inference model is a standalone model, with its own
encoder and network, sharing no trunk or weights with another model. From the observer's view, it
reads:

- **the current position:** the board, the current cross-checks, the
  observer's rack, the unseen pool, scores and the bag count;
- **the last 5 moves:** the opponent's last 3 and the observer's last 2,
  interleaved in game order. For each move, the squares it placed tiles on;
  the cross-checks as they stood *before* it was played, so the model sees
  which hooks and lanes the player had and passed up; and its tiles played,
  whether it was an exchange or a pass, its score and the bag count then;
- **the opponent's forgone plays:** for each of their last 3 moves and each
  tile kind `τ`, how much more the best play needing `τ` (the move's tiles
  plus one more `τ`) would have scored. The network cannot search the lexicon
  for this itself, and a large margin says the leave probably lacks `τ`
  ([deep_bot_v0.md](deep_bot_v0.md#forgone-plays)).

**The output: 29 floats.**

- **`θ_l`, one per letter (27):** a log-weight for "the leave contains `l`".
  Positive means likely has it, negative means likely doesn't, so one number
  covers both reads.
- **`a` and `b ≥ 0`:** a tilt on the leave's absolute quality `v(ℓ)`, its
  static leave value in equity points.

Those numbers define the leave distribution as the pool prior, tilted:

`log p(ℓ) = log P_prior(ℓ | pool) + Σ_{l ∈ ℓ} θ_l + a·v(ℓ) − b·v(ℓ)² − log Z`

over leaves `ℓ` of the public length that the observer's unseen pool can
supply. `P_prior` is the hypergeometric draw from that pool: `Π_l C(n_l, c_l)`
up to a constant, where `n_l` is the pool's count of `l` and `c_l` the
leave's.

- **`a = b = 0` is exactly "no information":** the leave is a uniform draw
  from the pool. `a > 0` shifts belief toward better leaves, `a < 0` toward
  worse ones, and `b > 0` concentrates it around a value.
- **The quality term carries the interactions.** Leave value already folds in
  balance, synergy, duplication and stems, so one tilt moves a whole class of
  leaves at once: raising `a` favors well-balanced, bingo-prone leaves and
  leaves unbalanced ones down. The letter terms stay per-letter, which is all
  the has-it and doesn't-have-it reads need.
- **Quality is absolute.** The tilt acts on leave value in points, against
  whatever the current pool can produce. With both blanks unseen, a positive
  `a` reaches up to blank leaves; with both played, it favors the best leaves
  that remain.

**The leave tables.** Static, built once offline, one per length from 1 to 6.
Each row is a leave: its 27 counts, a 27-bit presence mask, and `v(ℓ)`. The
6-tile table has 737,311 rows (about 30 MB); the 4-tile table has 25,254. A
7-tile leave arises only from a pass and is sampled from the prior alone. If
leave values become board-aware, only the `v(ℓ)` column changes, per position.

**Likelihood and training.** One pass over the table for the leave's length
gives the exact log-likelihood:

1. **Per-letter prior terms:** `log C(n_l, c)` for each letter and count, at
   most 27 × 8 values.
2. **Each row's log-weight:** skip the row if some `c_l > n_l`; otherwise
   `Σ_l logC[l][c_l] + Σ_{l ∈ mask} θ_l + a·v − b·v²`, about 28 additions.
3. **`log Z`** is the log-sum-exp of the log-weights, and the loss is
   `log Z` minus the true leave's log-weight.

Training directly on this likelihood matters. Per-letter has-it classifiers
and a separate quality regressor, trained independently, would give marginals
that are not the weights to multiply together. Trained jointly, the outputs
*are* the weights, and the readable per-letter probabilities ("likely has an
S") come out of the same pass as marginals, along with the implied mean and
spread of the opponent's leave value.

**Sampling.**

- **Algorithm 1: enumerate, then inverse CDF.** Compute every row's
  log-weight, exponentiate against the maximum, and take a running sum in the
  table's fixed order. For each sample's variate `u ∈ [0, 1)`, binary-search
  the CDF for `u × total`; that row is the leave. Then add the refill. A
  6-tile query is about 20M additions, 10–20 ms on one CPU thread, and each
  sample after that is a binary search. Because the order is fixed and
  sampling inverts a CDF, one variate gives the same leave whenever two
  queries have the same distribution, so samples pair across candidates. The
  same pass computes `log Z`, so training, evaluation and sampling share one
  implementation.
- **Algorithm 2: per-letter proposal, then rejection.** Drop the quality
  factor. What remains, `Π_l C(n_l, c_l) · exp(θ_l [c_l > 0])` at fixed length,
  factors over letters, so a small table of partial sums over (letter, tiles
  still to place) gives exact sampling by walking the letters. Accept with
  probability `exp(a·v − b·v² − M)`, where `M` bounds the quality term over
  that length's values. Accepted samples are exact. When the tilt is strong
  the acceptance rate falls, so the sampler estimates it from a few trial
  draws and falls back to Algorithm 1 below about 5%. Pairing is looser,
  since a varying number of variates is consumed.

| Use | Volume | Algorithm |
|---|---|---|
| Policy-set label generation | one query per position, ~300 samples | 1, on the CPU |
| The agent, Bob's beliefs | ~2,000 queries per turn, ~20 samples each | 1, batched on the GPU |
| Many unbatched CPU queries | — | 2, falling back to 1 |

**Conditioning on Alice's candidate.** Bob's belief about Alice is the leave inference
model run from Bob's view, with Alice's candidate `a` as the most recent move of
its history. That is one forward pass per `(a, r)`, batched on the GPU.

### Generations

A frozen policy `P_k` generates games, and those games train `I_k`, the
`k`-th leave inference model.

- **The labels are free.** A `.slog` stores the initial racks and every move
  with the tiles drawn after it, so replay reconstructs every player's true
  leave at every turn ([architecture.md](../architecture.md)).
- **Train `I_k` only on `P_k`'s games, warm-started from `I_{k−1}`.** Trained
  on a buffer that mixes generations, an inference network converges to Bayes'
  rule under the *average* policy, not the current one. Games for `I_k` are
  generated cheaply, by playing `P_k`'s policy-set model's top-1 without
  search.

`I_k` models opponents who play like `P_k`. Against Macondo or human players
that is a mismatch. Fine-tuning on recorded games with known racks is the
later remedy.

## Cost and allocation

Per Alice candidate:

| Item | Count |
|---|---|
| Policy-set calls for Bob | `R_A`: 100 |
| Bob's sims | `R_A · N_B · R_B`: 100 × 8 × 20 = 16,000 |
| Alice's sims | `R_A ·` sims per rack: about 400 |

Modeling Bob dominates, at about 40 to 1. With 20 candidates, a turn is
about 320,000 short sims. Each one is two mechanical moves plus a table
lookup, so move generation dominates: about 640,000 HastyBot moves per turn.
At roughly 0.1 to 0.3 ms each, that is 60 to 200 CPU-seconds, or very roughly
2 to 7 seconds on 28 threads. Four levers:

- **Successive halving over Alice's candidates.** A cheap first pass models
  Bob as the policy-set model's top-1. The full subtree runs only for
  survivors.
- **Racing in Bob's subtree.** Stop sampling Alice's racks once Bob's argmax
  is clear on the paired samples.
- **Caching `β(a, r)`.** Every replenishment sim of `(a, r)` reuses Bob's
  choice.
- **Batching** the network calls (the policy-set model's sets and the leave
  beliefs) across subtrees, following the `BatchingPositionEvalService`
  pattern.

## Build order

Each step names what it measures. The measurements guide the next step.
Steps 1 to 3 are planned in detail in [deep_bot_v0.md](deep_bot_v0.md).

1. **The leave inference model, `I_0`,** from HastyBot games, and the v0
   win table. *Measure:* held-out log-likelihood of the true leave against
   uniform draws.
2. **Policy-set labels.** Pools over positions from the whole game except the
   endgame, with opponent leaves from the leave inference model and rollouts
   stopped at the win table.
3. **Policy-set v0, both forms.** *Measure:* recall, meaning how often the
   pool's sim-best move is in `S`, against the top `N` by value, by form, bag
   size and `N`, with setups and exchanges reported separately; and each
   form's time per batch of sets.
4. **Bob's subtree, offline.** At known positions, for Bob's sampled racks
   after each of Alice's top candidates: Bob's chosen replies, how often
   each kind (bingo, setup, other) is chosen, and how stable the choice is
   in `R_B`.
5. **The agent.**
   - The seeded draw sequence, the sim with win-table leaves, pairing,
     two-level evaluation with inferred-leave beliefs, and the cache.
   - *Measure:* the picks at the known positions
     (`positions/NWL23/interesting-positions/`, the blind_spots tag), turn
     time, and matches against BestBot and `mset-sim` in standard Scrabble.
6. **Allocation.** Successive halving and racing. *Measure:* the strength
   lost against the time saved.
7. **Generations.** DeepBot labels positions for the next policy-set model,
   its games train the next leave inference model and fit the next win table.

## Limits

- **Recall is unrecoverable at search time.** If the policy-set model omits
  Bob's best reply, nothing in the search brings it back. Policy-set recall is
  therefore the first measurement. Online transfer between Bob's subtrees
  could be added if recall turns out to be the bottleneck.
- **Depth.** After the two searched decisions, every move is mechanical.
  Setups that pay off later than the sim's horizon are valued only by the
  win table. In Bob's sims, Alice's reply is hasty, so Bob's model of
  Alice does not look for blocks.
- **Bob's argmax.** A deterministic model of Bob, over paired means of about
  20 samples, is slightly biased toward moves that drew lucky samples.
  Scrabble gives little reason for mixed strategies, so v1 keeps the argmax.
  Mixing (a softmax, or sampling between the top two) is the easy change if
  Alice's model of Bob proves brittle.
- **The win table is a coarse leaf.** It sees the spread, the side to move's
  leave value and the tiles unseen, not the board: a wide-open board and a
  closed one read the same. A trained leaf network is the later
  replacement.

## Where this sits

- **[simulation_information_flow.md](../simulation_information_flow.md).**
  DeepBot's online search is local in that document's vocabulary: each
  candidate and each Bob decision is valued by its own sims. What the
  idealized agent there learns across racks and candidates, DeepBot learns
  offline into the policy-set model and the inference network. Recognizing
  which racks have a setup in a given spot is the policy-set model's job,
  trained across many positions, not something rediscovered in each search.
- **Macondo's BestBot** runs the same outer loop: candidates, sampled
  opponent racks, sims. Its opponent plays the static-equity argmax, so he
  never tries a setup that equity ranks low, and his rack is drawn
  uniformly. DeepBot replaces that
  opponent with a decision made on the opponent's own information, over a
  diverse candidate set.
- **Many-tree information-set search** (Cowling, Powley & Whitehouse, 2015)
  is the source of the rule that each player decides over their own beliefs,
  and that only moves cross between players' evaluations.
- **[sim_residual_feedback.md](sim_residual_feedback.md)** built the
  evidence-fusion stage and the proves-best head. The policy-set model's set
  conditioning and gain head are the same machinery, conditioned on chosen
  moves.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)** supplies the
  exhaustive survey recipe the policy-set model's labels extend.
- **[design.md §3](../design.md)** describes a public belief system. The
  generational inference network is a concrete first form of it.
