# DeepBot: two-level simulation with a policy-set model

**Status: proposed 2026-10-14; revised 2026-10-25 to build the agent first
on static-equity candidates ([Candidates](#candidates)); revised 2026-11-02
to carry each sim's evidence across the search ([Evidence across the
search](#evidence-across-the-search), [Budget](#budget)); revised
2026-11-03 to make the search network one model that reads the evidence
store, with end-of-game heads and a root correction head ([The search
network](#the-search-network)); revised 2026-11-03 again to keep each
player's evidence on its own side, fix the store's items, put a model-free
budget first and record the reply-model ablation ([The evidence
store](#the-evidence-store), [Budget](#budget), [Candidates](#candidates)),
with the policy-set model's set forms moved to an
[appendix](#appendix-the-shelved-set-forms); not yet placed in
the [roadmap](../roadmap.md).** DeepBot plays standard Scrabble, with hidden
leaves. This document is a design to guide development. Its measurements
inform the next step; no pass or kill thresholds are fixed in advance.

Write **Alice** for the player to move and **Bob** for the opponent.

## The turn

1. **Candidates.** The [proposer](#candidates) proposes Alice's first
   candidates from all her legal moves. More are
   [admitted](#steering) as the search finds evidence that they may be
   better than the leader.
2. **Bob's racks.** Alice samples `R_A` racks for Bob (about 100) from her
   [belief](#beliefs). They are fixed for the turn and shared by every
   candidate.
3. **Bob's move.** For each candidate `a` and each Bob rack `r`, Alice
   predicts Bob's reply `β(a, r)` by building [Bob's
   subtree](#bobs-subtree), which models Bob's decision on Bob's
   information. A candidate that [empties the
   bag](#when-alices-move-empties-the-bag) skips the subtree. The
   [budget](#budget) decides how many sims each rack's subtree gets; a rack
   whose reply the evidence already settles gets none.
4. **Alice's sims.** For each `(a, r)`, Alice runs a dozen
   [sims](#sims), each with different replenishment draws, split among
   Bob's contending replies `b` by their chance of being his choice
   ([Budget](#budget)). Each sim:
   1. Bob plays `b`.
   2. Alice and Bob each play one more move by a mechanical rule.
   3. The leaf is evaluated.

   The mean over `b`'s draws is her payoff `q_A(a, r, b)`. Her value `q(a)`
   is the mean over racks of her payoff at Bob's reply: at `β(a, r)`, or
   averaged over his contenders by their chance of being his choice where
   his subtree leaves them [tied](#budget).
5. **Evidence.** Every sim, at both levels, is compared with what the
   [search network](#the-search-network) predicted for it. The gaps are
   collected for the turn in an [evidence store](#the-evidence-store), each
   player's apart, and the search network reads them to predict the moves
   not yet simmed, under every candidate. They decide which moves get sims
   next, where sims stop, and Bob's choice at each rack.
6. **The pick.** Alice plays the candidate with the highest `q(a)`.

## Bob's subtree

Bob's subtree answers one question: given that Alice played `a`, and Bob
holds `r`, what does Bob play?

1. **Alice's racks, from Bob's side.** Bob samples `R_B` racks for Alice
   (about 20) from *his* belief. That belief conditions on the public history
   *including `a`*, and its pool is the tiles Bob cannot see, given `r`.
2. **Bob's candidates.** The [proposer](#candidates), evaluated from Bob's
   information set (`r` plus the public state), proposes his first moves.
   Others are [admitted](#steering) when the turn's evidence lifts them:
   a setup found for one of Bob's racks, under one of Alice's candidates,
   is tried for the other racks that can play it.
3. **Bob's sims.** For each candidate `b` and each of Alice's sampled racks:
   1. Bob plays `b`.
   2. Alice and Bob each play one more move by the mechanical rule.
   3. The leaf is evaluated from Bob's side.

   All of Bob's candidates use the same `R_B` racks and draws, so they are
   compared on paired samples.
4. **The choice.** `score(b)` is the mean over Alice's sampled racks. Bob's
   move `β(a, r)` is the top of the [search network's
   policy](#the-search-network) at `(a, r)` once it has read the turn's
   evidence, these scores among it ([The evidence
   store](#the-evidence-store)): the argmax of `score(b)` alone separates
   close moves poorly at `R_B` racks. Where Bob's leading moves stay within
   noise of each other and Alice's payoffs after them differ, the
   [budget](#budget) keeps them all, weighted by their chance of being
   best, instead of paying to separate them.

**Only the move crosses back.** Alice's sims use `β(a, r)`, never Bob's
scores. Bob's scores are expectations over Bob's belief about Alice's rack,
not over Alice's actual rack, so they belong to Bob's decision and nowhere
else. The [evidence store](#the-evidence-store) keeps the same rule: what
Bob's sims found informs Bob's decisions at other racks and candidates,
never Alice's values, and what hers found never his.

![Two-level simulation: Alice's evaluation spawns a subtree for Bob at each sampled rack, and only Bob's chosen move crosses back](../images/deep_bot_many_tree.svg)

*Each triangle is one player's evaluation, colored by whose beliefs it runs
on: red for Alice's, blue for Bob's. Each node is colored by whose turn it is,
and labeled with Alice's tiles, then Bob's, with `?` for tiles unknown to the
evaluation's owner.*

*Alice starts at `x ?`. She holds `x` and cannot see Bob's rack. The
proposer proposes `a` among her candidates. Alice samples Bob's rack
from her own belief, `y` or `z`. At `x z` it is Bob's turn, and Alice cannot
choose Bob's move with her own values, so she **spawns** Bob's subtree, rooted
at `? z`: Bob holds `z` and does not know Alice's tiles.*

*In that subtree, the proposer proposes Bob's set, `b₁`, `b₂`, `b₃`
and others. Bob samples racks for Alice from his own belief, which reads what
`a` reveals (`x̂₁`, `x̂₂`, …). For each of his candidates, he sims: Alice
plays hasty, Bob plays hasty, and the leaf network values the leaf. Each
candidate scores the mean over Alice's sampled racks, and Bob's move is the
argmax, here `b₁`.*

*Only that move crosses back. In Alice's evaluation Bob plays `b₁`; Alice
samples replenishments, both players continue with hasty moves, and the leaf
values average into `q(a)`. Every sampled rack gets its own subtree for Bob
(`y` at right).*

### When Alice's move empties the bag

If `a` draws the bag's last tiles, Bob's turn is an endgame. The tiles he
cannot see are exactly Alice's rack, and his subtree does not apply:

- the policy-set model trains on the whole game except the endgame, so its
  set at an empty bag is outside its training, and the equity proposer's
  recall was measured on the same positions;
- Bob needs no belief about Alice's rack, since he can deduce it, and Alice
  needs no replenishment draws, since there are none.

So for each sampled rack `r`, Alice skips the subtree and plays one sim to
the game's end: Bob and Alice alternate moves by the mechanical rule until
someone goes out or the game stalls, and the exact outcome enters `q(a)`.
This is the play-to-the-end rule of [the sims](#sims), started one move
earlier.

That makes Bob's empty-bag move the one decision in the design that he does
not search. The endgame solver (`EndgameSolver`) could choose it instead.
Measured 2026-10-24 on the first empty-bag position of 200 HastyBot games,
against a reference solve at budget 51,200 (which proved 131 of them):

| On the 131 proven positions | Matches the reference's class |
|-----------------------------|-------------------------------|
| Hasty to the end | 123 |
| The budget-400 solve's move, then hasty to the end | 126 |
| The budget-400 solve's own value, where it proved nothing (103) | 24 |

A budget-400 solve takes 2.4 ms on one thread, a hasty playout 0.34 ms. A
solve that proves nothing gives a usable move but not a usable value. The
solver's move is a later upgrade, worth what an A/B of the agent shows; the
positions above are mostly decided endgames, and the solver's gains in the
[endgame benchmark](../endgame_bench_results.md) sit in close ones.

## Sims

**The mechanical rule** is HastyBot's: static equity over the mover's own
legal moves. It reads only the mover's rack, so the moves it makes cannot
depend on the opponent's hidden tiles. In the whole design, only two
decisions search: Alice's at the root and Bob's in his subtree. Each is made
on its own player's information.

**The leaf: a network.** After a sim's fixed moves, a small network turns
the leaf position into a win probability: the
[leaf eval model](../deep_bot_leaf_eval_results.md), a conv trunk over the
board, the cross-checks, both players' leaves, the pool, the spread and the
pool size.

- **Both leaves are known to the leaf,** because the sim holds both racks:
  the mover's own, and the opponent's as sampled from the mover's belief. The
  network's value at one sampled leaf is not the value of the position as the
  mover sees it; the mean over the sampled leaves is.
- **The value is from the player who just moved.** It is taken from the side
  that owns the evaluation (Bob in his subtree, Alice in hers), as `1 − P`
  when the other side moved last.
- **If the game ends** inside the sim, the outcome is exact.
- **If the bag is empty** after the two mechanical moves, both players keep
  moving by the mechanical rule to the game's end, and the outcome is exact.
  The leaf network trains only on turns that began with tiles in the bag,
  and with both racks known and nothing to draw, the playout is the exact
  value a network would only approximate.

**A sim also returns maps of the board's future.** Besides its value, a sim
reports, for every square empty after its fixed moves, the
[search network's continuation targets](#the-search-network): who places a
tile there, whether that move is a bingo, and the signed points of that
move. Squares filled during the sim's mechanical moves are known exactly.
Squares still empty at the leaf take the leaf network's predictions, which
it makes with the same ownership and points heads. The sim also reports
who goes out and the net end-of-game rack adjustment: exact if the game
ended inside it, otherwise the leaf network's going-out and end-of-game
points heads predict them. That is how the sim's value is completed too:
what the sim played is exact, and the leaf network estimates the rest.

**The leaf network is versioned.** It is trained on `.slog` outcomes of the
current policy by the `deep_bot_leaf_eval` workload, which exports each
generation as an ONNX file stamped with its format version and the leave
values it read. The win probability of a leaf shifts as play gets stronger,
so each policy generation gets a new network, and everything it values
records the network's tag, generation and format version.

**A win table stays as a reference.** `P(win | spread + leave value, tiles
unseen)`, fitted from the same outcomes in the CSV format the engine reads
([sim/win_pct_table.h](../../engine/include/sim/win_pct_table.h)), is the
baseline the leaf network is measured against, and the leaf of the CPU sims
that predate the GPU path. It sees neither the board nor the opponent's leave.

**Sims run on the GPU.** A batch of sims is one kernel launch over thousands
of rollouts, one thread block each, with the lexicon tables resident on the
device ([gpu_hasty_rollouts.md](gpu_hasty_rollouts.md)). The kernel plays
every mechanical move, leaves each rollout's end state in device memory, and
a second kernel writes the leaf network's input rows straight into the
TensorRT engine's input buffers. Only three floats per rollout come back,
plus the maps above summed on the device over each candidate's rollouts, since
a full map per rollout would dwarf the rest of the transfer. A CPU reference
implements the same sim move for move, and the kernel matches it rollout
for rollout.

Every sim in the design has the kernel's shape: a position in which one
player has just moved, then the two mechanical moves, then the leaf from that
player's side (or, with the bag empty, the game played to its end).

- **Bob's sims** start after his candidate `b`; Alice and Bob reply hastily.
- **Alice's sims** start after Bob's reply `β(a, r)`, which the host applies
  to the board, since it is a fixed move. The position is then one in which
  Bob has just moved and Alice's leave is known, and the value is flipped to
  Alice's side.
- **Label sims** start after the candidate; the opponent and the chooser
  reply hastily.

The demo runs one position at a time. One launch is to take
([deep_bot_v0.md](deep_bot_v0.md#build-order), step 9):

- **many positions, in groups that share an unseen pool:** a group is the
  candidates of one chooser at one position, each applied to the board (a
  pool's candidates, Bob's set at one `(a, r)`, or Alice's sims of one
  `(a, r)`), and a launch holds as many groups as fill it;
- **per-rollout opponent leaves,** sampled on the host from the mover's
  belief, or known (Alice's own leave, in her sims); the kernel draws the
  rest of the rack;
- **a range of rollout indices** per group, so racing can run in rounds.

**Racing runs in rounds.** A round is one launch: a slice of the rollouts
for every candidate still in the race, then the host compares the candidates
on their paired samples and drops the clearly beaten ones before the next
round. Rollout `i` is the same deal in whichever round it runs. Rounds from
many positions and subtrees share each launch, so a launch stays large while
each race is narrow.

**Pairing.** Samples come from variates keyed by `(sample index, draw
ordinal)`:

- **Bob's racks** come from Alice's unseen pool, which does not depend on her
  candidate. So every candidate meets exactly the same `R_A` racks, and
  comparisons between candidates are paired.
- **Within Bob's subtree,** all of his candidates share the same `R_B` racks
  for Alice and the same draws.
- **Replenishments** come from each player's own end of one bag per sample,
  shuffled once: Alice draws from one end and Bob from the other, so each
  player's k-th tile is the same whichever candidate she played and however
  many tiles either of them draws. In Alice's sims after Bob's reply, that
  bag is the bag as she found it (her leave seated, her exchanged tiles
  returned after her draw), the same for every candidate, so her candidates
  stay paired on Bob's draws too.

A seeded, indexable draw sequence replaces the `Game`'s bag in sim code: the
GPU kernel's counter-based generator (Philox, keyed by the batch seed and
counted by rollout index), which the CPU reference shares. Rollout `i`
shuffles the same pool the same way in every start of a group.

## Candidates

**The proposer.** Given a position from the chooser's information set and a
size `N`, the proposer returns `N` of the chooser's legal moves. Alice's
first candidates and Bob's first moves at each rack both come from it. The
set is where the search starts, not where it ends: [steering](#steering)
admits the moves the turn's evidence lifts.

**v0 proposes the static-equity top `N`.** The learned forms of the
[policy-set model](#the-policy-set-model) so far match that baseline and do
not beat it, so the agent is built on it, and the policy-set model replaces it
once it beats equity at the same `N`. The equity proposer needs no network:
one move generation per set, on the CPU. Bob needs a set per `(a, r)`, about
2,000 per turn, each well under a millisecond.

**`N` depends on the stage of the game.** Measured 2026-10-25 on 2,000 pools
labeled on the GPU ([Labels](#labels): every legal move, Bob's reply hasty,
the leaf network at the end). A pool's best move is its largest `q`. Regret
is the best move's win probability minus that of the set's own best-`q`
move, in points of win probability:

| `N` | Recall, all | Regret, turns 0–5 | Regret, turns 6–11 | Recall, turns ≥ 12 | Regret, turns ≥ 12 | 95th-percentile regret, turns ≥ 12 |
|---|---|---|---|---|---|---|
| 4 | 0.800 | 0.13 | 0.14 | 0.679 | 1.35 | 6.5 |
| 8 | 0.866 | 0.06 | 0.06 | 0.750 | 0.98 | 4.9 |
| 16 | 0.905 | 0.03 | 0.05 | 0.815 | 0.79 | 3.8 |
| 24 | 0.928 | 0.02 | 0.01 | 0.852 | 0.58 | 2.7 |
| 40 | 0.944 | 0.01 | 0.01 | 0.878 | 0.42 | 1.6 |

Turn counts both players' moves; the pools record no bag size. Each turn
bucket holds 600 to 800 pools. Through turn 11 the top 8 lose almost nothing.
From turn 12 on, the top 8 miss the best move a quarter of the time, and even
the top 40 lose 0.4 points on average. So `N` grows with the stage, about 8
early and 24 to 40 late, which makes late turns several times dearer in Bob's
subtree. The late game is also where a learned proposer has room to beat
equity.

**The reply model.** How Alice models Bob's move is a knob:

- **`subtree`:** Bob's subtree, as designed.
- **`hasty`:** Bob plays the static-equity argmax for `r`, which is Macondo
  BestBot's model of the opponent, here with the leave inference beliefs and
  the leaf network. The subtree is skipped, which makes a turn about 40 times
  cheaper ([Cost and allocation](#cost-and-allocation)).

- **`alice-subtree`:** Bob plays HastyBot's move, and the search moves to
  Alice's second turn. At each `(a, r)` she holds the rack her refill drew,
  samples Bob's leave from her belief after his reply, and sims her own
  static-equity top moves. The move she picks then gets two hasty plies and
  the leaf network, which value `a`. Against a HastyBot opponent the reply
  model is exact, so what this setting measures is Alice modeling her own
  later play as a searcher rather than as HastyBot. `--second-candidates 1`
  keeps the depth (four hasty plies to the leaf) without her search: the
  control for what depth alone is worth.

`hasty` is the first pass of successive halving, and the two settings at
equal time measure what modeling Bob on his own information is worth.
Against HastyBot itself that comparison cannot favor the subtree, since
`hasty` is then Bob's exact model and the subtree can only add noise to it.
Measured 2026-10-28 (the `match_arms` tags `ablate-*`), 1,000 paired games
per arm against HastyBot, every arm's confidence half-width about 0.024:

| Arm | Score | Match time, s |
|---|---|---|
| `subtree`, the defaults | 0.605 | 6,789 |
| `hasty` | 0.595 | 394 |
| Alice's candidates halved / doubled | 0.575 / 0.605 | 5,633 / 6,927 |
| Alice's racks 10 / 40 | 0.599 / 0.592 | 2,602 / 12,814 |
| Bob's racks 50 / 200 | 0.596 / 0.584 | 3,417 / 13,482 |
| Bob's candidates halved / doubled | 0.594 / 0.588 | 4,291 / 11,472 |
| Candidates reaching the subtree 4 / 16 | 0.580 / 0.590 | 3,468 / 11,299 |
| Sims per rack 2 | 0.585 | 6,706 |
| Uniform racks for Bob, no leave inference | 0.588 | 6,027 |
| No endgame solver | 0.589 | 6,794 |

Every arm lies within one interval of the defaults, and the subtree costs
17 times `hasty`. That is the null the setting guarantees: the arms say the
subtree does not hurt, and nothing more. The belief that it matters rests
on tournament positions in which modeling the opponent as hasty misjudges
the line, above all a setup only the mover's rack can exploit, and such
positions are not rare.

**The match that can show the subtree's worth** is against an opponent the
subtree models well. DeepBot with the `hasty` reply model is one: Bob's
subtree models Bob as a one-level searcher who assumes a hasty Alice, and
that is what `hasty`-DeepBot is, one ply deeper. In that match the
subtree's model of its opponent is nearly exact, and the `hasty` model is
wrong where it matters. After a candidate that sets up a hook only Alice's
rack can fill, `hasty`-DeepBot as Bob blocks once his belief gives her the
tile enough weight, since his sims see her mechanical continuation fill it;
`hasty`-Alice assumes he never blocks and overvalues the setup, while
`subtree`-Alice predicts when he does. Macondo's BestBot is the other such
opponent, and dearer to run. On the ablation's timings the match costs
about the `subtree` arm's, two hours per thousand games on the dev laptop.

Whole games are a blunt instrument for it: under both reply models most
turns pick the same move and contribute only noise. The sharp one plays
out the disagreements. `--subtree-dump` records the turns of self-play
where the two reply models pick differently; from each such position both
picks are played out against the opponent many times on paired draws, so
every sample lands on a disagreement. `alice-subtree` remains the setting
for what Alice modeling her own later play as a searcher is worth.

## Evidence across the search

Bob's subtrees dominate a turn's cost ([Cost and
allocation](#cost-and-allocation)), and run independently they are weak as
well as dear:

- **One subtree cannot separate close moves.** In the policy-set pools, the
  per-rollout difference between two moves spreads 11 to 16 points of win
  probability, against typical margins of about 1 point. At `R_B = 20`, the
  standard error of a difference between two of Bob's moves is about 3
  points, and separating a 1-point margin would take on the order of 700
  samples. Measured 2026-10-27, rerunning a subtree at `R_B = 20` chose the
  same reply 35 to 55% of the time.
- **Every subtree rediscovers the same lessons.** Bob's racks are the same
  under every candidate, and any one candidate leaves most of the board
  untouched. A setup that pays for one rack is open to every rack that can
  play it, under every candidate that leaves its lane alone.

So a sim's result is also evidence about the rest of the tree. The evidence
is the gap, the **residual**, between what a sim found and what a network
predicted it would find. One network carries it: the search network makes
the predictions with no evidence, the sims' residuals against them are
stored for the turn, and the same network, reading that store, predicts the
moves not yet simmed.

### The search network

One network serves both levels. Given a position from the mover's
information set, it predicts what the sims after each of the mover's moves
will measure, and which move a one-level search would choose. It reads the
board, the cross-checks, the unseen pool, the bag count, the scores, the
mover's own rack and the mover's [belief](#beliefs) about the opponent's
rack, summarized per letter. It builds on the move set model's trunk and
move encoder, so the board is encoded once per position and each move adds
only its own features.

The **evaluator** is the player whose evaluation the sims belong to: Bob in
his subtree, Alice in hers. Per-player targets are written from the
evaluator's side.

**The store is an optional input.** A store-attention block over the
trunk's output reads the turn's [evidence store](#the-evidence-store), the
items of the evaluator's own side. With an empty store the block is
skipped, and the outputs are the network's prior; with evidence, the value
and policy outputs are corrected by it. The trunk runs once per position,
when the search first reaches it, and the store block runs again as
evidence arrives, against the cached trunk output.

**Continuation heads.** These predict what follows a fixed move (or moves)
when play continues as the [sims](#sims) play it: the mechanical moves, then
the leaf.

- **Value:** the evaluator's win probability, as a mean and a variance.
- **Ownership:** for every empty square, five classes: the evaluator places
  a tile there in a bingo, the evaluator places one in another move, the
  opponent does either, or the square stays empty to the game's end.
- **Points:** for every empty square, the expected signed points of the move
  that places a tile there (positive for the evaluator, negative for the
  opponent, zero if it stays empty), and their variance.
- **Going out:** three classes: the evaluator goes out, the opponent does,
  or nobody does (the game ends on scoreless turns).
- **End-of-game points:** the net end-of-game rack adjustment from the
  evaluator's side, as a mean and a variance. When a player goes out, it is
  what that player collects from the other's rack, signed by who did; when
  nobody does, it is the opponent's remaining rack value minus the
  evaluator's.

Why the targets have this shape:

- **The bingo split costs nothing.** The five classes sum to the three that
  say only who places the tile. A bingo lane is open to a whole class of
  racks, the bingo-prone ones, rather than to particular tiles, and bingos
  carry a timing effect of their own (seven tiles turned over), so the
  split gives a direct read that the points cannot.
- **Points are the whole move's, not a share.** A move scores as a unit,
  and a square's danger is what the move through it scores: a hook
  dangerous because of a 63-point bingo along it is a 63-point square, not
  a 9-point one. Unlike Go's ownership, the points map therefore does not
  sum to the score. Go places one stone per turn; a Scrabble move places
  one to seven tiles, so a share per tile would dilute the large moves.
  The value head carries the score.
- **Points are unconditional, with variance.** Every sim gives every square
  a residual, with an empty square counting zero. A head conditioned on a
  class would get evidence only when that class occurred, which for a
  rare class on a given square is seldom. The variance separates a likely
  small move from an unlikely large one with the same mean; the ownership
  classes add who would make it.
- **The end of the game is timing, not place.** Near the end, who goes out
  first, and what the other player is left holding, can decide the game
  while the board barely changes, and the maps do not show it. The
  end-of-game points are unconditional for the same reason as the squares':
  a game nobody goes out of still has a net adjustment, so every game labels
  them.
- **Losses fit moments, not a Gaussian.** The per-square points target is a
  spike at zero plus scattered large values. The mean trains on squared
  error, and the variance on squared error against the squared deviation
  from the predicted mean, with no gradient through that mean. Each is
  consistent whatever the target's shape, and the end-of-game points use the
  same losses. In the transfer test, plain Gaussian likelihood let a head
  lower its loss by widening its spread instead of moving its mean
  ([supreme_bot_m1a.md](supreme_bot_m1a.md)).

**The policy head.** `π(m | position, mover's rack, mover's belief)`: the
probability that `m` is the mover's best move by a **one-level search**.
That search sims each of the mover's moves with the opponent's rack drawn
from the mover's belief, the opponent and then the mover replying hastily,
and the leaf network at the end; the best move has the largest mean. Only
the mover's own move is searched, so it is the mover's best against a hasty
opponent. The pools label exactly that ([Labels](#labels)), so the head is
the [policy form](#the-policy) of the policy-set model, trained on pools
from self-play positions, from the mover's side.

DeepBot uses it at Bob's level, where it is the right question: Bob's
subtree is that search, with Alice replying hastily, so `π` predicts the
move his subtree would choose with unlimited sims. There is no model of Bob
as such. Alice poses the question from his side: his rack `r`, and his
belief about her given `a`, read through the reply view that hides his
private past ([Beliefs](#beliefs)), as the leave inference model trains to.
At each `(a, r)`, the policy:

- orders his moves for sims ([Steering](#steering));
- is his reply at a rack the [budget](#budget) gives no sims, as a
  distribution, so that a near-tie does not snap to one move;
- is his choice at a rack that has sims, once it has read them
  ([The evidence store](#the-evidence-store));
- weights his replies in Alice's value for a candidate, below.

Alice's decision at the root is made by search, from values, and does not
read `π`. At her position, `π` would answer "her best move against a hasty
Bob", which is the question the subtree is there to improve on.

Two choices shape the head:

- **A policy, not the value head's chance of being best.** Moves' sim
  outcomes are strongly correlated through shared racks and draws: the
  per-rollout spread is far larger than the margins between moves
  ([Evidence across the search](#evidence-across-the-search)). Per-move
  means and variances cannot recover the chance that a move is best; a head
  trained on the pools' paired chance of best can.
- **One level, by design.** Bob models Alice as hasty, which is where the
  design's recursion stops. A Bob who models Alice as a searcher would be a
  further level of subtrees, a setting of the [reply model](#candidates),
  not a head.

This head is where the searching Bob lives in the network. Every value in
the design is a continuation after fixed moves; searching changes only which
moves are fixed. So Alice's prediction for a candidate decomposes as

`q̂(a) = mean_r Σ_b π(b | a, r) · V(a, b)`

with `V` the value head from Alice's side after `a` and `b`.

**The root correction head.** At Alice's position, the value head values
each of her moves against a hasty Bob, as the pools label it. What modeling
Bob as a searcher changes is `Δ(a) = q_subtree(a) − q_hasty(a)`: her value
for `a` with Bob's subtree as the [reply model](#candidates), minus her
value with `hasty`. The head predicts `Δ` as a mean and a variance, and
`V(a) + Δ̂(a)` is her prior for `q(a)`, which [steering](#steering) uses to
admit and order candidates before their children are evaluated. After that,
`q̂(a)` is composed from the children, so the head reads no store.

- **Its labels come from DeepBot's own turns.** Successive halving values
  every candidate with `hasty` first and runs the subtree for the survivors
  ([Budget](#budget)), over the same Bob racks, so each survivor's `Δ` is a
  paired difference, less noisy than either value alone. Self-play logs it
  per candidate with its sample count, and the loss is masked to the logged
  candidates.
- **A value correction, not a two-level policy.** The root admits
  candidates by expected improvement, from means and variances, so
  `V + Δ̂` and its spread carry what Alice needs. A kind of move DeepBot has
  not searched keeps a correction near zero, and its one-level value
  stands, where a two-level policy would extrapolate freely.

**Maps are computed only for simmed moves.** A residual map needs the
predicted map of the post-move board, which is a spatial decode per
candidate. Moves that get sims need it to form their residuals; moves that
don't need only the network's scalar outputs.

**v0: the leaf network predicts.** The leaf network already values a
position right after a fixed move under the hasty continuation, and gets
the other continuation heads too ([Sims](#sims)). Until the search network
is built, a simmed move's prediction is the leaf network's on the post-move
position, averaged over the racks the sims sampled. The residual is then
the difference between the network's value after two mechanical moves and
its value before them, as in temporal-difference learning: what those two
moves revealed that the network did not see.

**Training data.** Four sources, each exact for what it labels:

- **Game records.** A `.slog` replay knows who placed every tile, in which
  move, for how many points, whether the move was a bingo, who went out,
  the end-of-game rack adjustment, and who won. The continuation heads
  pretrain on that for free, as KataGo trains its ownership head on finished
  games.
- **Pools.** Self-play positions with every legal move simmed from the
  mover's side ([Labels](#labels)) train the policy and value heads with
  an empty store.
- **Grouped pools** ([below](#the-evidence-store)) train the value and
  policy outputs with a store.
- **DeepBot's root records** train the root correction head.

**Coverage grows with generations.** Bob decides after Alice's candidates,
including setups and exchanges that self-play rarely plays, so at generation
0 (HastyBot self-play) the positions after them are thin in the data. Each
generation's self-play comes from a stronger agent, which plays such moves
when they win, so the positions after them fill in, as in AlphaZero.

### Residuals

Every sim yields, against the search network's prior for its fixed moves:

- **a value residual:** the sim's value minus the predicted value;
- **map residuals:** for each square, the realized class minus the predicted
  class probabilities, and the realized signed points minus the predicted
  mean (and the squared deviation minus the predicted variance);
- **end-of-game residuals:** who went out minus the predicted classes, and
  the net rack adjustment minus its predicted mean (and the squared
  deviation minus the predicted variance). A sim that stops at the leaf
  completes them with the leaf network's predictions, as it completes the
  maps.

At Bob's level, each rack also yields a **reply residual**: his subtree's
chance-of-best weights from its sims, against `π`. That is the lesson
"after this candidate, a rack like this one replies there", and it is keyed
by the squares the reply uses, which is what lets it travel.

**Residuals are taken against the prior.** The prediction is always the
network's empty-store output, never its store-informed one, so an item
means the same thing whatever enters the store later.

### The evidence store

The **evidence store** holds the turn's residuals, in two kinds of item.

- **A map item** is one per evaluator, candidate `a` and move `b` (Bob's
  reply at his level, `β(a, r)` at Alice's): the map and end-of-game
  residuals summed over every sim of that pair, with the sim count. The
  board after `a` and `b` is the same whichever rack played `b`, so the
  sims of every rack share one map, which is also the shape the kernel
  returns ([Sims](#sims)). It records the squares each move places tiles
  on and the cross-check changes it makes.
- **A scalar item** is one per evaluator, `a`, rack `r` and `b`: the value
  residual, at Bob's level the reply residual, the leave of the rack that
  moved, and the context after `a`: the bag count and the mover's belief
  summary about the opponent's rack. It is the part that says whether a
  rack can use what the map shows.

There is one store per turn, not one per candidate. At the counts of
[Cost and allocation](#cost-and-allocation) it holds up to
`N_A · R_A · N_B` scalar items at Bob's level, about 16,000, with Alice's
`N_A · R_A` beside them, and `N_A` times the replies distinct across racks
in map items, which can run to hundreds per candidate; the
[budget](#budget) makes both fewer.

**Each side's items stay on their side.** Alice's sims run with her actual
rack; Bob's run with her rack sampled from his belief. The two carry
different information, and a target reads only the items of its own
evaluator. Suppose Alice's candidate sets up a hook that only a tile in her
rack can fill. Her items show the hook's squares as hers, with high points,
because her mechanical continuation fills them in every sim. Bob's items
show them as weakly hers, because from his side she holds the tile a
minority of the time. A Bob-level target that could read her items would
learn what she holds, which the real Bob cannot know, block the hook, and
leave Alice undervaluing every setup she considers; an Alice-level target
that read his would pull her value toward his uninformed view. The grouped
pools ([below](#the-evidence-store)) train the block the same way, since
each group is one mover's evidence, so a store that mixed sides when
serving would also lie outside its training.

**The search network reads it.** For a target, one of Bob's moves at one
rack under any candidate, the store-attention block attends over the map
items of the target's side, each carrying its scalar items pooled over
their racks by a set encoder that keeps each rack's leave, so that a lane
paying only for racks that hold its hook letter is read as such. At the
counts above that is thousands of map items, the regime the M1a reader
lost its way in ([Where this sits](#where-this-sits)), so region is a
filter before it is a match: a target attends to the items whose squares
or hooks it touches, whose reply its candidate leaves intact, or that
belong to its own candidate, since end-of-game evidence is timing rather
than place. The filter is a bitboard test on the host, and it leaves the
dozens of items that share the target's region. The value output becomes
a corrected mean and variance, and the policy a corrected distribution
over the rack's moves. Their gap from the empty-store pass is the
correction; with no evidence it is zero, so a search starts from the
prior. The ownership, points and end-of-game heads stay store-free: they
exist to form residuals. The block attends by two kinds of match:

- **Region:** whether the target places tiles on, or changes the hooks of,
  the squares an item's residuals concern, and whether the target's
  candidate leaves intact the squares and hooks an item's reply used. A
  candidate that blocks a lane cuts off the evidence that lane produced.
- **Context:** whether the item's bag count, leave and belief match the
  target's. A setup whose payoff depends on the opponent being unable to
  go out after a bingo pays when two tiles remain in the bag and not when
  five do. Whether a rack can use a lane depends on its tiles: a hook only
  an S fills pays for a rack holding the S, or likely to draw it. So both
  sides' context is an input.

**A rack's own sims are evidence too.** Once some of Bob's moves at
`(a, r)` have sims, their residuals are in the store, so the policy at that
rack combines its own sims with every other rack's evidence. Its top move
is Bob's choice there, in place of the argmax of his sims alone, which
separates close moves poorly at `R_B = 20` ([Evidence across the
search](#evidence-across-the-search)). The policy's target is the choice his
subtree would make with unlimited sims, and the rest of the turn's evidence
is what brings a few sims closer to it.

**Evidence crosses candidates.** Bob's racks are shared by every candidate,
and any one candidate leaves most of the board unchanged. A reply found
under one candidate therefore informs Bob's racks under every candidate
that leaves its region intact in a similar context. Most of the redundancy
between subtrees lies there.

**Training data: grouped pools.** The store input learns from the same kind
of decision Bob's subtrees make, a player to move choosing by hasty-reply
sims, grouped the way a turn's evidence is. A generator on the pool labeler
takes a self-play position, the parent, and several of the moves available
there; at each resulting position it samples several racks for the player
to move, and labels each rack's pool. It writes:

- per (move from the parent, rack, reply): the value residual and the
  reply's chance of being best, a scalar item's content;
- per (move from the parent, reply): one map, averaged over the racks that
  played the reply, a map item's. The map describes the board after both
  moves, which is the same for every such rack; whether a rack can use it
  is the per-rack scalars' part.

Maps are written only for each rack's leading replies, the ones a search
would sim, in a sidecar beside the pools. Moves, racks and replies per
group are the generator's parameters, and they set the size of a group's
record.

**The evidence has a search's noise.** A search's store holds a few
rollouts per move, and only for the moves it admitted; a pool labels every
legal move with many. So the visible evidence is a subsample of each pool's
paired rollouts, at the counts a search runs and only for each rack's
leading moves, while the targets are the full labels. Maps are evidence
only, never targets, so the generator writes each one over rollout slices
of those sizes. Trained on clean, complete evidence, the network would trust
the noisy, partial evidence of a search too much.

**Training: masked prediction on the grouped pools.** Hide whole moves from
the parent, or racks within one, and predict their full-label value
residuals and chances of being best from the rest of the group. A hidden
rack keeps a subsample of its own rollouts visible, from none to a search's
full count, so the network learns the whole range from no sims to many;
with everything hidden, the target is the empty-store prior. That is the
store input's task in search, including the cases where transfer should
fail (a lane blocked, a different bag count). The value loss is a
likelihood (beta-NLL, as in the transfer test), so the predicted variance is
calibrated: the [budget](#budget) relies on it.

**Generations.** Each generation's search network trains on positions from
the previous agent's self-play, and its root correction head on that
agent's root records. Over generations the prior absorbs the common
lessons, and the store input is left with what is new in each position.

### Steering

A move is admitted for sims by its prospects against the current leader:

- **at Bob's level,** by its chance of being his best at the rack, from the
  store-informed policy;
- **at the root,** by its expected improvement over Alice's leader, from her
  prior `V + Δ̂` and its spread until the candidate's children are
  evaluated, and from `q̂(a)` after.

A move the evidence lifts enters whatever the proposer thought of it; a
move predicted to be far below the leader is never simmed. A small floor on
admission keeps a wrong prediction from going uncorrected forever: a move
that is never simmed never produces the evidence that would correct it.

## The policy-set model

**Interface.** Given a position from the chooser's information set, all legal
moves and a size `N`, return a set `S` of `N` moves.

**The chooser's belief is an input.** The model also reads the chooser's
belief about the opponent's rack, from the
[leave inference model](#the-leave-inference-model), summarized per letter.
The labels are simmed with opponent racks drawn from that belief, so without
it the model would have to re-infer it from the board alone, with no move
history and no forgone plays. With it, a setup the opponent's last move left
can be read as "he likely holds the hook letter", and blocking it valued
accordingly. The chooser always has this belief: at Alice's root it is hers
about Bob, and in Bob's subtree it is his about Alice given `a` and his
sampled rack `r`, the belief his sims already draw from.

The opposite belief, the opponent's about the chooser's rack, is left out:
at Alice's root it needs a model that infers from public information alone,
since Alice does not know Bob's rack, and current labels could not teach its
use, since rollouts never read a belief.

**Objective.** Choose `S` to maximize the expected best value in the set,
`E[max_{a∈S} v(a)]`, where `v` is the move's value under the chooser's
uncertainty about it. This differs from taking the `N` moves most likely to
be best one at a time: a set of near-identical moves adds little over its
best member, so the objective rewards diversity. Of a strong play and four
near-copies of it, the set is better with one copy replaced by a different
kind of move, such as a setup that wins in other worlds.

Two set forms were built for that objective, a greedy selector with cached
steps and a one-shot slot decoder, and a third form, a policy, sets
diversity aside. All build on the move set model's architecture (the board
trunk, the move encoder and the cross-check encoder), and all train on the
same labeled pools ([Labels](#labels)). The set forms are shelved
([Appendix](#appendix-the-shelved-set-forms)): both tied the static-equity
top `N` on recall, the greedy form's predicted gains did not depend on the
set it was given, and the diversity term they exist for, which only regret
rewards, is small in Scrabble when measured on stored pools: skipping a
member whose rollouts track an earlier member's deal for deal left regret
unchanged (0.721 vs. 0.722 points at `N = 8`). The policy is the form in
use.

### The policy

The network gives each legal move one logit, and a softmax over the
position's moves, however many it has, makes a policy `π`. The loss is the
cross-entropy against `q`, `−Σ_a q(a) log π(a)`, and the set of `N` is the
policy's top `N`.

The events "`a` is the best move" are disjoint, so `Σ_{a∈S} q(a)` is the
probability that `S` holds the best move. A policy that matched `q` would
therefore make the best set for recall and cover at every `N`, from one
pass. DeepBot builds many sets per turn, one per `(a, r)` in Bob's
subtrees alone, and each network call is one batched call over all the
sets in progress. The policy is the [search network's](#the-search-network)
policy head. Measured 2026-10-29 on 2,000 pools it tied equity on recall;
with 6,800 training pools it beat equity on regret at `N = 4` (0.52
against 0.61 points of win probability). It underfits its own training
pools, so the lever is the model, not more data.

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
  [sim_labeled_candidates.md](sim_labeled_candidates.md) (every legal play)
  produces this shape of data.
- **The labels can see moves the rollout policy never plays.** Every
  candidate is simmed directly, rather than reached by a rollout policy that
  would have to choose it. A setup that static equity ranks low is labeled by
  its own outcome, provided its payoff arrives within the sim's horizon.
- **The target: a soft best move.** `q(a)` is the probability that `a`
  is best under the paired sim statistics: by bootstrapping over the paired
  samples, or by a normal approximation to the paired differences against
  the leader. Near-equal moves share the mass, so the policy learns that
  either will do. Racing stops weak moves early, on fewer rollouts, and both
  estimates account for that, since a move far below the leader rarely comes
  out on top. The greedy form's targets, assembled from subsets of the pool,
  are in the [appendix](#appendix-the-shelved-set-forms).

**Label rollouts stop at the leaf network.** Played to the end of the game,
a rollout is slow and its outcome noisy. A label rollout instead plays the
two mechanical replies on the GPU and ends at the [leaf network](#sims),
whose identity the pool records.

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
The leave itself is drawn from the **leave pool**: the unseen pool less any
tiles the observer exchanged just before the opponent's last move. The
opponent's rack predates that exchange, though the refill may hold its tiles.
The leave's length is public: the opponent's rack size minus the tiles their
last move played or exchanged (a pass keeps the whole rack).

**Inputs.** The leave inference model is a standalone model, with its own
encoder and network, sharing no trunk or weights with another model. From the observer's view, it
reads:

- **the current position:** the board, the current cross-checks, the
  observer's rack, the unseen pool, scores and the bag count, and the tiles
  the observer exchanged on each of their last two turns, if they did;
- **the opponent's last 3 moves.** The observer's own moves say nothing about
  the opponent's leave beyond the board, except those exchanges. For each move, the squares it placed tiles on;
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

over leaves `ℓ` of the public length that the leave pool can supply. `P_prior` is the hypergeometric draw from that pool: `Π_l C(n_l, c_l)`
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
its history. That is one forward pass per `(a, r)`, batched on the GPU
(`belief::ReplyView`).

**Bob's private past is hidden.** The row reads the observer's own past: the
tiles they exchanged on their last two turns, and their racks at the time of
the older history moves, through those slots' leave-kind masks. Inside Alice's
model, Bob's real past is not hers to see, and it would contradict the rack `r`
she sampled for him. So Bob's row hides it (`LeaveHistory::hide_private`): no
exchanges, the older slots' masks open to every kind, nothing subtracted from
the leave pool for his exchange, and an input flag that says so. The newest
slot, `a` itself, keeps a mask read from `r` and that pool.

Training hides it too, in a fraction of rows (`hide_private_fraction`, 0.25 by
default), so the model learns both readings rather than meeting the hidden one
only at serving time, as KataGo trains on positions with their history removed
for analysis without a game record. Each generation's held-out eval scores the
same rows both ways and charts the gap (`eval_hidden_gap`), so a model that
learns to lean on the private past shows it. Before the flag existed, a model
trained without hidden rows lost +0.0001 ± 0.0006 nats to the hidden form on
tilted2's held-out rows, and on the 2% of rows whose observer had just
exchanged, zeroing the exchange alone improved the NLL by 0.048 ± 0.014 nats.

### Generations

A frozen policy `P_k` generates games, and those games train `I_k`, the
`k`-th leave inference model.

- **The labels are free.** A `.slog` stores the initial racks and every move
  with the tiles drawn after it, so replay reconstructs every player's true
  leave at every turn ([architecture.md](../architecture.md)).
- **Train `I_k` only on `P_k`'s games, warm-started from `I_{k−1}`.** Trained
  on a buffer that mixes generations, an inference network converges to Bayes'
  rule under the *average* policy, not the current one. Once a policy-set
  model serves, games for `I_k` are generated cheaply, by playing `P_k`'s
  policy-set model's top-1 without search. Under the equity proposer that
  top-1 is HastyBot, the policy `I_0` already models, so games come from the
  agent itself at a reduced budget.

`I_k` models opponents who play like `P_k`. Against Macondo or human players
that is a mismatch. Fine-tuning on recorded games with known racks is the
later remedy.

## Cost and allocation

Per Alice candidate, with fixed counts:

| Item | Count |
|---|---|
| Candidate sets for Bob | `R_A`: 100 |
| Bob's sims | `R_A · N_B · R_B`: 100 × 8 × 20 = 16,000 |
| Alice's sims | `R_A ·` sims per rack: 100 × 12 = 1,200 |

Modeling Bob dominates, at about 13 to 1, and scales with `N_B`, which the
equity proposer needs larger late in the game ([Candidates](#candidates)).
With 20 candidates, a turn is
about 340,000 short sims, each two mechanical moves after a fixed one, and a
leaf network evaluation. On the dev laptop's GPU (RTX 5000 Ada), the rollout
kernel runs 140,000 to 540,000 two-ply rollouts a second, depending on how
many racks hold a blank, and the leaf network about 270,000 evaluations a
second in FP16; together, 90,000 to 170,000 sims a second
([gpu_hasty_rollouts.md](gpu_hasty_rollouts.md#the-leaf-network-on-the-device)).
That is roughly 2 to 4 seconds a turn. The same sims on the 28-thread CPU,
with the win table as the leaf, ran 20,000 to 164,000 rollouts a second in the
same measurement, so 2 to 17 seconds a turn; with the leaf network fed from
the host, the input upload bounds them at about 48,000 evaluations a second.

**The search network's passes are not in the table.** Each of Bob's
positions `(a, r)` is one trunk pass, `N_A · R_A` of them a turn, about
2,000 with the counts above, and the store block runs again each racing
round against the cached trunk output, with a spatial decode for each move
that gets sims. On the policy-set model's slot form, which shares the
trunk family, 256 sets of 8 took 38 ms on the same GPU, so a turn's
positions cost a few hundred milliseconds a pass, and a few passes a turn
sit well under the sims' 2 to 4 seconds. Step 10 of the [build
order](#build-order) measures the actual figure.

**The counts are CPU-sized.** On the GPU a launch wants 10,000 rollouts or
more to fill the device, and a turn's time is its number of racing rounds
times a round's launch, since every subtree's round shares the launch. Width
(`N_A`, `N_B`, `R_A`, `R_B`) costs throughput, not rounds, so the GPU may
favor wider sets and more racks than the counts above, with fewer rounds.
The round schedule is re-tuned there by measurement.

### Budget

Fixed counts give every `(a, r)` the same `N_B · R_B` sims, whatever the
turn has already learned. With the [evidence](#evidence-across-the-search),
each of Bob's moves at each rack carries a belief about its value to him:
the search network's store-informed value, with its variance, and its
store-informed policy, both updated as the move's own sims enter the store.
Three decisions read those beliefs.

1. **Which of Bob's moves get sims.** At each rack, the moves with a real
   chance of being his best, which replaces a fixed `N_B`
   ([Steering](#steering)).
2. **When a rack stops.** When one move's chance of being best is near
   certain, or when Alice does not care which of his contenders he plays:
   her payoffs `q_A(a, r, b)` after them barely differ. At fixed counts
   she runs a dozen sims per rack against his `N_B · R_B` (about 160), so
   simming after each of his contenders costs her little, and her payoffs
   say where his choice matters. Where his contenders stay within
   noise of each other and her payoffs after them differ, more sims cannot
   settle it at any affordable count (about 700 samples for a 1-point
   margin, [above](#evidence-across-the-search)). Her payoff there is the
   mean over his contenders, weighted by their chance of being best.
3. **Which racks get sims at all.** A rack whose reply the beliefs already
   settle before any sims gets none, and its `β(a, r)` is the predicted
   move. Where the policy is no sharper than static equity, which is where
   its form stands today ([The policy](#the-policy)), that move is close to
   the equity argmax, and the rack has fallen back to the `hasty` reply
   model. A small random share of such racks is simmed anyway, as an audit.
   An audit's sims enter the store like any others, and where they
   disagree with the prediction, a larger share of the settled racks is
   simmed for the rest of the turn.

Decisions 2 and 3 choose Bob's reply from predictions in place of sims.
Only the move crosses back, so a wrong prediction picks a wrong reply for
that rack, while Alice's payoff after it still comes from her own sims.

**The payoff test in decision 2 and the bound at the root need no network,
and come first.** Alice's payoffs after Bob's contenders are her own sims,
the rule that stops a rack when they barely differ reads nothing else, and
his contenders' chances of being best come from his sims' paired statistics
where they stay close, as the [labels](#labels) compute them. Built on the
fixed-count agent with successive halving ([Build order](#build-order),
step 6), they measure how much of the subtree's cost the turn's own sims
can shed before any model is trained, and set the floor the learned budget
is measured against.

**Alice's sims at a rack go to Bob's moves by their chance of being his
choice.** Each of the rack's sims in turn goes to the move with the largest
`w_b / (2 n_b + 1)`, where `w_b` is the move's chance of being best and
`n_b` the sims it already holds (the Sainte-Laguë rule), the lower equity
rank on ties. The moves that get any are his contenders, and her payoff at
the rack is their mean weighted by `w_b` renormalized over them:

- **Proportional is the split her value wants.** Contender `b`'s sims
  estimate a term of her value weighted by `w_b`, so for equally noisy sims
  the split that minimizes the variance of the weighted mean gives each
  contender sims in proportion to `w_b`. An equal split after each contender
  spends most of its sims on terms that weigh little: at chances 0.8, 0.1,
  0.1 and 12 sims, 4 each leaves her payoff about twice the variance of
  10, 1, 1. Concentrating the sims on `β(a, r)` matches that variance but
  keeps the error of the argmax snapping a near-tie, which more of her sims
  never shrink.
- **A likelier move never gets fewer sims, and one more sim only adds
  one.** Cumulative rounding of the chances loses the first (0.31, 0.29,
  0.25, 0.15 at ten sims gives 4, 2, 3, 1; the rule gives 3, 3, 3, 1). The
  second is what racing her candidates needs: a candidate stopped early
  holds the split a smaller count would have given.
- **A move without sims drops out.** It is a move whose share rounds to
  nothing, which replaces a floor on the chance and a cap on the count;
  the weight it carried shrinks with the rack's count.
- **Sim `j` of each contender deals the same draws.** The rack's
  replenishment is keyed by rack and sim index, the same for every
  candidate of Alice's and every move of Bob's, so the differences between
  his contenders stay paired over the sims they share, and every candidate
  meets the same draws. One unlucky bag is one sim at one rack.

The split serves her value, not the payoff test: where a contender holds
one sim, its gap to `β(a, r)` rests on one paired sample. The test tops up
the contenders whose gap could stop the rack, which is the allocator's
call.

**A second audit covers the racks that have sims.** Bob's choice there is
the store-informed policy's top move, not the argmax of his sims ([The
evidence store](#the-evidence-store)), so a prior that is merely weak, read
through a store with little to add, would draw his choice back toward the
prior's top move over what his own sims found, and nothing in the turn
would show it. At a small share of simmed racks the subtree runs on to a
high sample count, and the store-informed choice is compared with the
choice those sims make alone. Decision 3's audit catches a prediction that
is wrong; this one catches a prior that overrides the sims.

**At the root,** each candidate's payoffs bound what any model of Bob could
make of it: `q(a)` lies between `mean_r min_b q_A(a, r, b)` and
`mean_r max_b q_A(a, r, b)` over Bob's contenders. A candidate whose upper
bound is below the leader's lower bound needs no more of Bob's sims.

**One allocator.** All of a turn's sims go through one queue, and each
racing round is one GPU launch built from its top. A sim of Bob's move is
worth roughly the chance it changes his choice at that rack, times how much
that changes Alice's payoff, times the chance its candidate still
contends. Alice's sims and her candidates' first passes join the same
queue.

Three cheaper levers remain:

- **Successive halving over Alice's candidates.** A first pass models Bob
  as the proposer's top-1 (the `hasty` reply model). The subtree runs only
  for candidates that survive it.
- **Caching `β(a, r)`.** Every replenishment sim of `(a, r)` reuses Bob's
  choice.
- **Batching** the network calls (the leave beliefs and the search
  network's two passes) across subtrees, following the
  `BatchingPositionEvalService` pattern.

## Build order

Each step names what it measures. The measurements guide the next step.
Steps 1 to 3 are planned in detail in [deep_bot_v0.md](deep_bot_v0.md).
Steps 4 to 6 run on the equity proposer, so step 3 does not gate them.

1. **The leave inference model, `I_0`,** from HastyBot games, and the v0
   win table. *Measure:* held-out log-likelihood of the true leave against
   uniform draws.
2. **Policy-set labels.** Pools over positions from the whole game except the
   endgame, with opponent leaves from the leave inference model and rollouts
   run on the GPU, stopped at the leaf network.
3. **Policy-set v0.** Built: the set forms tied the equity proposer and
   are shelved ([Appendix](#appendix-the-shelved-set-forms)); the policy
   form goes on as the search network's policy head (step 10), and
   replaces the equity proposer once it beats it at the `N` the stage
   needs. *Measure:* recall, meaning how often the pool's sim-best move is
   in `S`, against the top `N` by value, by bag size and `N`, with setups
   and exchanges reported separately. The equity baseline's curve is in
   [Candidates](#candidates).
4. **Bob's subtree, offline.** At known positions, for Bob's sampled racks
   after each of Alice's top candidates: Bob's chosen replies, how often
   each kind (bingo, setup, other) is chosen, and how stable the choice is
   in `R_B`. DeepBot's `--subtree-dump` records these per turn in play
   (agent/subtree_dump.h), in place of a separate tool.
5. **The agent.**
   - The GPU sims with groups and racing rounds, the leaf network, pairing,
     two-level evaluation with inferred-leave beliefs batched on the GPU, and
     the cache.
   - The proposer's `N` by stage of the game, and the reply-model knob.
   - *Measure:* the picks at the known positions
     (`positions/NWL23/interesting-positions/`, the blind_spots tag), turn
     time, and the subtree against the `hasty` reply model: the two
     DeepBots against each other, and each pick played out from the
     self-play turns where they disagree ([Candidates](#candidates)); then
     matches against BestBot and `mset-sim` in standard Scrabble. The
     ablation against HastyBot is done, and null by design.
6. **Allocation without a model.** Done: successive halving and racing,
   and Alice's sims at each rack split among Bob's contenders by their
   chance of being best from his own sims, which gives the
   [budget](#budget) its payoff vector. The rest of the model-free budget,
   a rack that stops when her payoffs after Bob's contenders barely differ,
   the bound at the root, and racing her candidates partway through the
   split's sequence, saves time without changing what the agent computes,
   and its ceiling is modest beside the learned budget's, so it moves to
   step 12, whose decisions need Bob's subtree to pause mid-turn in any
   case.
7. **Continuation heads on the leaf network,** trained from game records:
   ownership and points, then going out and end-of-game points.
   *Measure:* held-out loss per head by bag size, the maps at the known
   positions (whether a hook the mover alone can fill, or a setup beside an
   existing hook, shows in the maps), and the going-out calibration in the
   last turns.
8. **Maps from sims.** The kernel and the CPU reference return each
   candidate's summed maps and end-of-game outcomes, completed by the leaf
   network's predictions, and the agent forms residuals against the leaf
   network's prediction on the post-move position (v0, [The search
   network](#the-search-network)). *Measure:* the noise of a square's
   residual against the rollout count. Done; measured on DeepBot's
   after-reply sims under the `v1` leaf network, 3 games each at 4, 12 and
   48 sims per rack (`py/scripts/deep_bot/residual_noise.py`), per start
   summed over its ~180 empty squares: the points residual's squared signal
   is about 1,300 points² (2.7 points rms per square) at every rollout
   count, and its sampling noise about 23,000 / n, so the noise falls below
   the signal between 12 and 24 sims, at 12 (the agent's) it is 1.6x the
   signal and at 48 0.4x. The ownership residual's signal is within its
   noise at every count (about 1 summed squared share against 1.0 at 12):
   two plies move the prior's ownership little. The end-of-game points
   residual carries a signal of about 8 points² against a noise of 0.7 at
   12 sims. The prior's variance is calibrated: it equals the rollouts'
   spread plus the leaf's variance at every count, and its value agrees
   with the sims' mean to 0.002.
9. **Grouped pools.** The generator on the pool labeler, as a dashboard
   workload, writing the scalars and the search-noise maps beside the
   pools. *Measure:* bytes and GPU time per group.
10. **The search network with an empty store:** the continuation heads,
    pretrained on game records, and the policy and value heads, trained on
    self-play pools from the mover's side. *Measure:* how often the
    policy's top move is the pool's best, against equity's
    ([Candidates](#candidates)), and how often it is the subtree's choice
    at the agent's positions.
11. **The store input,** trained by masked prediction on the grouped pools.
    *Measure:* its value and policy predictions against the empty-store
    pass, split by evidence from the same candidate and from other
    candidates; the calibration of its variance; and, at a rack with a
    search's count of its own sims, how often its top move is the rack's
    full-label best, against the argmax of those sims alone.
12. **Steering and the learned budget in the agent:** admission, budget
    decisions 1 and 3 with their audit, Bob's choice from the
    store-informed policy with the simmed-rack audit, and the single
    allocator, with the model-free stops deferred from step 6: a rack stops
    when her payoffs after Bob's contenders barely differ, a candidate whose
    best case falls below another's worst case is dropped, and racing her
    candidates stops each one partway through the split's sequence.
    *Measure:* sims per turn and strength at equal time against the agent
    that sims every rack in full, and both audits' disagreement rates.
13. **Generations.** DeepBot's self-play logs each turn's paired `Δ` for
    the candidates that reach the subtree. Its games, its root records, and
    the pools and grouped pools labeled from its positions train the next
    leave inference model, leaf network and search network, the root
    correction head included. *Measure:* the root correction head's
    held-out error against `Δ = 0`, and how often the subtree's pick is
    among Alice's first candidates, with the head and without.

## Limits

- **First discovery.** [Steering](#steering) spreads a move kind once some
  rack's sims have found it, but one of them must be simmed first. That
  rests on the proposer, the search network's prediction and the admission
  floor. A move kind none of them ranks is found only by the floor, and
  late in the game, where the equity proposer's recall is weakest
  ([Candidates](#candidates)), that matters most.
- **The store input can be wrong.** In steering, a wrong correction
  wastes sims. At a rack the [budget](#budget) skips, it picks a wrong reply
  for Bob, and the audits bound how often that goes unnoticed. At a rack
  with sims, it can outweigh them, less so as they accumulate. Evidence that
  should not transfer (a different bag count, a rack without the hook
  letter) is what the context inputs are for, and the masked training
  includes those cases.
- **The maps see places, not timing.** Some moves win by what they leave
  in the bag, not by any square: whether a bingo leaves the opponent able
  to go out. The going-out and end-of-game points heads read that near the
  end of the game; earlier, the value residual carries it.
- **Depth.** After the two searched decisions, every move is mechanical.
  Setups that pay off later than the sim's horizon are valued only by the
  leaf network. In Bob's sims, Alice's reply is hasty, so Bob's model of
  Alice does not look for blocks.
- **Bob's choice under noise.** An argmax over paired means of about 20
  samples is biased toward moves that drew lucky samples. Bob's choice is
  the store-informed policy's top move instead, which reads the rest of the
  turn's evidence; its bias is the prior's and the evidence's, which the
  grouped pools' full labels train against, and the [budget](#budget)'s
  simmed-rack audit measures whether it has collapsed to the prior's alone.
  Where his contenders are still within noise and Alice cares which he
  plays, the [budget](#budget) weights them by their chance of being best.
- **The leaf network values the policy of its training games.** v0's
  targets are outcomes of games HastyBot finished, so it values a leaf as
  HastyBot would play it out: a position that only stronger play would
  exploit reads as HastyBot leaves it. Each generation retrains it on the
  current policy's games.

## Where this sits

- **[simulation_information_flow.md](../simulation_information_flow.md).**
  In that document's vocabulary, DeepBot steers sideways at the root and
  below it: one sim's residuals choose which moves are simmed elsewhere,
  across Bob's racks and across Alice's candidates. Alice's valuation stays
  local, each of her payoffs from her own sims; Bob's choice at a rack reads
  the whole turn's evidence, his own sims at that rack among it. Draws are
  fixed, and nothing is revised. What generalizes across many positions
  (which racks have a setup in a given spot) the search network's prior and
  the inference network learn offline; the search network's store input
  carries what is new in the position at hand.
- **[supreme_bot.md](supreme_bot.md)** proposed reading sim residuals
  against a prior with a learned network. Its transfer test found no
  transfer ([supreme_bot_m1a.md](supreme_bot_m1a.md)): the teacher prior
  already priced what hasty rollouts reveal, and the reader struggled to
  find a candidate's probes among about 2,000 tokens. Here the evidence is
  residual maps matched to a target by the squares it touches, against
  continuation heads trained to predict the sims themselves.
- **KataGo** (Wu, 2019) trains an ownership head on finished games as an
  auxiliary target. The continuation heads take that idea to Scrabble with
  two changes: the bingo classes, and per-square points of the whole move
  through the square rather than a share of the score, since a move places a
  varying number of tiles.
- **AlphaZero** trains its network on positions from its own self-play, with
  its search's results as targets. The search network does the same: the
  positions come from DeepBot's self-play, the policy's targets from the
  pools' one-level search, the search Bob's subtree runs, and the root
  correction head's from DeepBot's own root search.
- **Macondo's BestBot** runs the same outer loop: candidates, sampled
  opponent racks, sims. Its opponent plays the static-equity argmax, so he
  never tries a setup that equity ranks low, and his rack is drawn
  uniformly. DeepBot with the `hasty` reply model is that loop with inferred
  racks and the leaf network. With `subtree`, it replaces that opponent with
  a decision made on the opponent's own information.
- **Many-tree information-set search** (Cowling, Powley & Whitehouse, 2015)
  is the source of the rule that each player decides over their own beliefs,
  and that only moves cross between players' evaluations.
- **[sim_residual_feedback.md](sim_residual_feedback.md)** built the
  evidence-fusion stage and the proves-best head. The shelved greedy form's
  set conditioning and gain head ([Appendix](#appendix-the-shelved-set-forms))
  are the same machinery, conditioned on chosen moves.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)** supplies the
  exhaustive survey recipe the policy-set model's labels extend.
- **[design.md §3](../design.md)** describes a public belief system. The
  generational inference network is a concrete first form of it.

## Appendix: the shelved set forms

The two set forms of the [policy-set model](#the-policy-set-model), built
as the `policy_set` workload's `greedy` and `slot` forms and shelved
2026-10-29: on 2,000 pools, and over a 35-generation run, both tied the
static-equity top `N` on recall at `N = 8` and did not beat it on regret,
and the greedy form's predicted gains came out flat across the sets it was
given. The text stands as designed.

### Form 1: greedy selection, with cached steps

The expected max of a set of win probabilities is monotone and submodular, so
greedy construction comes within a factor of `1 − 1/e` of the best set. The
policy-set model picks one move at a time, each maximizing its predicted gain
over the moves already chosen:

`gain(a | S) = E[max(0, v(a) − max_{s∈S} v(s))]`

`S` starts from the seed, the static-equity top move, rather than empty. At
`S = ∅` the gain would be the value itself: a whole win probability, where
every later step's is a margin of a few points, so that one step would set
the gain loss's scale. Seeded, every step asks which move most improves on a
real one; the seed is the sim-best move about half the time, so a set of two
or more would hold it anyway. Greedy from a fixed seed keeps the guarantee
for the moves it adds.

The expectation is over what the sims have not resolved about the values,
not over the game's own luck. A training pool's sims leave each pair of
moves a mean difference with a standard error, so each member `s` of `S`
gives `a` an expected improvement under that uncertainty,
`E[max(0, μ_a − μ_s)]`. The target is the least of them. Since
`max_S μ ≥ μ_s` for every member, each is an upper bound on the gain over the
set's best, and the least is the tightest. It matters when `a` sims as a
member does: if `A` is the set's best and `B` a member `a` duplicates, then
`a` beats `A` only where `B` already does. Against `A` alone `a` would look
promising; against `B` its improvement is 0. The bound is loose only when the
set covers `a` jointly and no single member does, and then it errs toward
crediting `a`. The exact value (resampling the paired rollouts and comparing
`a` with the set's best on each resample) is the alternative if that proves
to matter.

An expectation over rollouts instead, `E_r[max(0, v_r(a) − v_r(s))]`, was
considered and rejected. It is half the mean margin plus half the mean
absolute per-rollout difference, and that per-rollout spread (11 to 16
points, inferred from the pools' paired standard errors) dwarfs typical
margins (about 1 point), so it would rank moves by volatility: a move 5
points worse than the set's best but volatile would outrank one 2 points
better.

A stage conditions each move's
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

**Targets: subset assembly.** Any subset `S` of the pool, plus a held-out
move `a`, gives a row whose target is the least expected improvement of `a`
over a member of `S`, computed from the paired rollout values. One pool
yields many rows.

### Many sets at once

DeepBot builds many sets per turn. Bob's subtrees alone need one per
`(a, r)`, about 100 per candidate. With either form, each network call is one
batched call over all the sets in progress. Form 1 makes `N` such calls, all
but the first of them light; Form 2 and the policy make one.
