# DeepBot: two-level simulation with a policy-set model

**Status: proposed 2026-10-14; not built, and not yet placed in the
[roadmap](../roadmap.md).** DeepBot plays standard Scrabble, with hidden
leaves. This document is a design to guide development. Its measurements
inform the next step; no pass or kill thresholds are fixed in advance.

Write **Alice** for the player to move and **Bob** for the opponent.

## The turn

1. **Candidates.** The [policy-set model](#the-policy-set-model) (PSM)
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
2. **Bob's candidates.** The PSM, evaluated from Bob's information set (`r`
   plus the public state), proposes `N_B` moves.
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

*Alice starts at `x ?`. She holds `x` and cannot see Bob's rack. The PSM
proposes `a` among her candidates. Alice samples Bob's rack from her own
belief, `y` or `z`. At `x z` it is Bob's turn, and Alice cannot choose Bob's
move with her own values, so she **spawns** Bob's subtree, rooted at `? z`:
Bob holds `z` and does not know Alice's tiles.*

*In that subtree, the PSM proposes Bob's set, `b₁`, `b₂`, `b₃` and others.
Bob samples racks for Alice from his own belief, which reads what `a` reveals
(`x̂₁`, `x̂₂`, …). For each of his candidates, he sims: Alice plays hasty, Bob
plays hasty, and the leaf is evaluated by the network, or by the solver once
the bag is empty. Each candidate scores the mean over Alice's sampled racks,
and Bob's move is the argmax, here `b₁`.*

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

**The leaf.**

- **Bag empty:** if the bag is empty when a player is to move, anywhere in a
  sim, the sim stops and the endgame solver values the position. Within a
  sampled world both racks are known, so the endgame is
  perfect-information. When a candidate and a reply empty the bag between
  them, every sim of that line ends in a solve, so endgame tactics the
  mechanical rule would misplay (a block, a two-way out) are valued exactly.
- **Otherwise,** after the sim's fixed moves, the position evaluation network
  reads the position from the side that owns the evaluation. Within a sampled
  world the opponent's leave is known, so today's face-up checkpoints can
  read it. They value the rest of the game as if leaves were public, a bias
  that lasts until a standard-Scrabble teacher is trained.

The solver runs in class mode. A proven class gives 1, ½ or 0. An unproven
solve falls back to the network's reading of the same position.

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

**Form: greedy selection with a set-conditioned gain head.** The expected max
of a set of win probabilities is monotone and submodular, so greedy
construction comes within a factor of `1 − 1/e` of the best set. The PSM picks one move at a
time, each maximizing its predicted gain over the moves already chosen:

`gain(a | S) = E[max(0, v(a) − max_{s∈S} v(s))]`

At `S = ∅` this reduces to ranking by value. The architecture is the move set
model's: the board trunk, the move encoder and the cross-check encoder, plus
a stage that conditions each move's query on the tokens of the moves already
chosen. The evidence-fusion stage built for
[sim_residual_feedback.md](sim_residual_feedback.md) already conditions on a
set of move tokens. There the tokens carried sim results; here they are
chosen moves alone. Its proves-best head is the same quantity with "simmed"
in place of "chosen".

**Labels.** A labeled position is a pool of its legal moves with values from
sims, run from the chooser's information set: the opponent's rack is drawn
from the chooser's belief, and the sims are paired across moves.

- **Rows by subset assembly.** Any subset `S` of the pool, plus a held-out
  move `a`, gives a row with target `max(0, v(a) − max_S v)`, computed from
  the paired values. One pool yields many rows.
- **Pools need the tail.** A pool must include moves the prior ranks low
  (exchanges, short plays, setups), or the PSM never learns that they can
  win. The exhaustive survey recipe of
  [sim_labeled_candidates.md](sim_labeled_candidates.md) (every legal play,
  screened at 1,000 rollouts, racing) produces this shape of data. Its
  existing outputs were run face-up and its rollouts are hasty throughout, so
  PSM labels need a standard-Scrabble mode.
- **The labels can see moves the rollout policy never plays.** Every
  candidate is simmed directly, rather than reached by a rollout policy that
  would have to choose it. A setup that static equity ranks low is labeled by
  its own outcome, provided its payoff arrives within the sim's horizon.

**Generations.** Once DeepBot runs, its own evaluations label positions, and
each PSM generation trains on the previous agent's labels, as the move
proposal model's generations do.

## Beliefs

Two beliefs feed the design: Alice's over Bob's rack (step 2), and Bob's over
Alice's rack given `a` (his subtree, step 1).

**First version: `RackInferrer`**
([belief/rack_inference.h](../../engine/include/belief/rack_inference.h)).
It infers the opponent's leave from their last move under a HastyBot-equity
model of how they choose, and adds a uniform refill. For Bob's belief, the
pool depends on Bob's sampled rack, so a naive implementation runs one
inference per `(a, r)`. It factorizes instead:

- the likelihood of `a` given a leave depends only on the board and the
  leave, so it is scored once per candidate;
- the hypergeometric prior depends on the pool, so it is reweighted per rack.

**Target: a generational inference network.** A frozen policy `P_k` generates
games, and those games train an inference network `I_k`.

- **The labels are free.** A `.slog` stores the initial racks and every move
  with the tiles drawn after it, so replay reconstructs every player's true
  rack at every turn ([architecture.md](../architecture.md)). A row is the
  observer's view of the public history, and the target is the opponent's
  true rack.
- **Train `I_k` only on `P_k`'s games, warm-started from `I_{k−1}`.** Trained
  on a buffer that mixes generations, an inference network converges to
  Bayes' rule under the *average* policy, not the current one. Games for
  `I_k` are generated cheaply, by playing `P_k`'s PSM top-1 without search.
- **Letter by letter.** The network predicts the rack autoregressively in
  canonical sorted order: a 27-way softmax per step, with letters the pool
  has run out of masked. That gives exact likelihoods, needs no enumeration,
  and samples by inverse CDF from a supplied variate, which keeps racks
  paired across candidates.
- **History as input.** The reads that matter span several moves: a short
  play after a scoring play can point to a bingo-prone rack, and a play the
  opponent passed up can show that a tile is still in the bag. The opponent's last few moves enter as tokens (tiles,
  exchanges, scores), which the transformer trunk takes naturally.
- **Conditioning on Alice's candidate** is one forward pass per `(a, r)`,
  batched on the GPU.

`I_k` models opponents who play like `P_k`. Against Macondo or human players
that is a mismatch. Fine-tuning on recorded games with known racks is the
later remedy.

## Cost and allocation

Per Alice candidate:

| Item | Count |
|---|---|
| PSM calls for Bob | `R_A`: 100 |
| Bob's sims | `R_A · N_B · R_B`: 100 × 8 × 20 = 16,000 |
| Alice's sims | `R_A ·` sims per rack: about 400 |

Modeling Bob dominates, at about 40 to 1. With 20 candidates, a turn is
about 320,000 short sims. Each one is two mechanical moves plus a batched
network evaluation or a solve, which puts a turn at very roughly 10 to 20
seconds on the development machine. Four levers:

- **Successive halving over Alice's candidates.** A cheap first pass models
  Bob as the PSM's top-1. The full subtree runs only for survivors.
- **Racing in Bob's subtree.** Stop sampling Alice's racks once Bob's argmax
  is clear on the paired samples.
- **Caching `β(a, r)`.** Every replenishment sim of `(a, r)` reuses Bob's
  choice.
- **Batching** network evaluations across sims and subtrees, following the
  `BatchingPositionEvalService` pattern.

## Build order

Each step names what it measures. The measurements guide the next step.

1. **PSM labels.** A standard-Scrabble mode for the exhaustive survey: the
   opponent's rack drawn from the chooser's belief, paired sims over every
   legal play, with solved endgames. Run it as a tag over self-play
   positions, weighted toward small bags, where the survey found plays
   outside the top 10 matter most.
2. **PSM v0.** The greedy gain head on the move set architecture, trained by
   subset assembly. *Measure:* recall, meaning how often the pool's sim-best
   move is in `S`, against the top `N` by value, by bag size and `N`. Recall
   of setups and exchanges, the moves static equity ranks low, is reported
   separately.
3. **Bob's subtree, offline.** At known positions, for Bob's sampled racks
   after each of Alice's top candidates: Bob's chosen replies, how often
   each kind (bingo, setup, other) is chosen, and how stable the choice is
   in `R_B`.
4. **The agent.**
   - The seeded draw sequence, the sim with solver leaves, pairing,
     two-level evaluation, `RackInferrer` factorized, and the cache.
   - *Measure:* the picks at the known positions
     (`positions/NWL23/interesting-positions/`, the blind_spots tag), turn
     time, and matches against BestBot and `mset-sim` in standard Scrabble.
5. **Allocation.** Successive halving and racing. *Measure:* the strength
   lost against the time saved.
6. **The inference network `I_0`,** from `P_0`'s games. *Measure:* held-out
   log-likelihood of the true rack against `RackInferrer` and uniform
   draws, by number of history moves. Then swap it into the agent.
7. **Generations.** DeepBot labels positions for the next PSM, its games
   train the next `I_k`, and a standard-Scrabble teacher replaces the
   face-up leaf model.

## Limits

- **Recall is unrecoverable at search time.** If the PSM omits Bob's best
  reply, nothing in the search brings it back. PSM recall is therefore the
  first measurement. Online transfer between Bob's subtrees could be added
  if recall turns out to be the bottleneck.
- **Depth.** After the two searched decisions, every move is mechanical.
  Setups that pay off later than the sim's horizon are valued only by the
  leaf network. In Bob's sims, Alice's reply is hasty, so Bob's model of
  Alice does not look for blocks.
- **Bob's argmax.** A deterministic model of Bob, over paired means of about
  20 samples, is slightly biased toward moves that drew lucky samples.
  Scrabble gives little reason for mixed strategies, so v1 keeps the argmax.
  Mixing (a softmax, or sampling between the top two) is the easy change if
  Alice's model of Bob proves brittle.
- **The face-up leaf model.** It values the rest of the game as if leaves
  were public, until the standard-Scrabble teacher exists.

## Where this sits

- **[simulation_information_flow.md](../simulation_information_flow.md).**
  DeepBot's online search is local in that document's vocabulary: each
  candidate and each Bob decision is valued by its own sims. What the
  idealized agent there learns across racks and candidates, DeepBot learns
  offline into the PSM and the inference network. Recognizing which racks
  have a setup in a given spot is the PSM's job, trained across many
  positions, not something rediscovered in each search.
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
  evidence-fusion stage and the proves-best head. The PSM's set conditioning
  and gain head are the same machinery, conditioned on chosen moves.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)** supplies the
  exhaustive survey recipe the PSM's labels extend.
- **[design.md §3](../design.md)** describes a public belief system. The
  generational inference network is a concrete first form of it.
