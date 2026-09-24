# Levels of simulation

**Purpose.** A vocabulary for comparing agents that decide by simulation, and a
ladder of levels ordered by how far the information from one simulation
travels through the game tree. Macondo's BestBot, the reference opponent this
project measures against ([evaluation_plan.md](evaluation_plan.md)), sits at
Level 1, the simplest scheme that simulates at all. Scribblez's search work
aims at the levels above it
([sim_residual_feedback.md](plans/sim_residual_feedback.md),
[rack_conditional_evidence.md](plans/rack_conditional_evidence.md)). Those
levels depend on a model that
[sim_labeled_candidates.md](plans/sim_labeled_candidates.md) sets out to fix.

The levels work like SAE's driving-automation levels, which were defined long
before any car could drive itself: they describe what a scheme does, not what
has been built.

## Simulation schemes

A simulating agent searches a game tree rooted at the current position. The
tree has two kinds of node:

- **Action nodes**, where a player chooses a move. The root is one, and its
  edges are the candidate moves.
- **Chance nodes**, where the game deals. Examples are the tiles drawn from
  the bag after a move, and the opponent's unknown rack, sampled at the start
  of every simulation (under face-up leaves, only the tiles the opponent
  drew are unknown).

Every such agent runs the same loop:

1. **Probe.** Walk from the root to a leaf, choosing an edge at every node: a
   move at action nodes, a draw at chance nodes.
2. **Record.** Read an observation at the leaf and add it, with the path, to
   the **record**. The observation may be the final spread, a win or a loss,
   or an evaluation of the horizon position. It can be richer, for example
   which replies were played and on which racks.
3. **Repeat** until the budget is spent or a stopping rule fires.
4. **Decide.** Score every root candidate from the record, and play the best.

Two things read the record:

- **Steering** is how the record shapes the next probe, that is, which edge
  the probe takes at each node.
- **Valuation** is how the record prices an edge. That covers the score step
  4 reads at the root, and any values the scheme keeps below the root.

A scheme is characterized by what each of the two may read.

Everything the agent knew before the first probe is its **prior**. That
includes static equity, leave values, a trained network, and a rack-inference
distribution computed from the opponent's last play. A prior can inform any
step, but it stays fixed during the search, so it carries nothing from one
probe to another. The levels below concern the record only.

## Where information goes

### Locality is decided at the edge

Write `S(e)` for the probes that crossed edge `e`. An estimate attached to `e`
is **local** if it reads only `S(e)`, and **global** if it also reads probes
that went elsewhere.

The obvious first attempt draws the line at the node instead: a decision at
node `n` is local if it reads only the probes that passed through `n`. That
fails at the root, where every probe passes through. Every root decision
would then be local, and the question that matters most would have no answer:
does simulating CAT say anything about DOG? Drawn at the edge, the line
answers it. An estimate of DOG that reads CAT's probes is global.

- **Valuation** is local or global according to the estimate it produces.
- **Steering** is **non-adaptive** when it ignores the record. A fixed
  rollout policy and draws from the prior's distribution are both
  non-adaptive.
- Otherwise steering has the scope of the edge estimates it compares. A
  bandit choosing among a node's children, or pruning against the leader, is
  local steering. Each option is judged by its own probes; comparing options
  is what steering does.

### Action nodes and chance nodes

Scope applies at both kinds of node. Taking Alice to move, with candidates CAT
and DOG and Bob replying, the four combinations ask different questions:

| | valuation | steering |
|---|---|---|
| action node | Does simulating CAT tell us how good DOG is? | Does a probe in which Bob draws AEI after CAT change whether we simulate DOG next? Does it change which reply Bob plays in DOG's rollouts? |
| chance node | Does Bob's result on the rack AEINRST after CAT tell us his result on AEINRSU? | Because racks with an F decided CAT, should Bob's racks after DOG be drawn with more F's? |

Chance nodes differ in one respect. An action node is worth its best edge, so
steering there is meant to concentrate on good edges. A chance node is worth
the expectation over its draws at their true probabilities. Steering there may
concentrate on informative draws, but valuation must then undo the skew with
importance weights. Otherwise the scheme values the draws it chose to look at,
not the ones the bag will deal.

### The profile

A scheme's **profile** records the scope of steering and valuation at three
places: the root, the action nodes below it, and the chance nodes.
[BestBot's](#bestbot-is-level-1) is below. The levels summarize a profile by
its farthest reach.

## The levels

| level | a probe's information reaches | examples |
|---|---|---|
| 0 | nowhere: there are no probes | static equity (HastyBot); one-pass model scoring |
| 1 | its own root candidate, as one sample of an average | Macondo's BestBot; Scribblez's `sim`, `neural-sim` and `mset-sim` agents |
| 2 | later probes through the same nodes | MCTS with UCT; AlphaZero |
| 3 | the values of branches it did not visit | shared transposition statistics; RAVE in Go; UltimateBot's evidence loop, at the root |
| 4 | the steering of probes in branches it did not visit | killer and history move ordering in chess; the forward transfer of rack-conditional evidence |
| 5 | the probes recorded before it | the re-pricing of rack-conditional evidence |

A scheme's level is the farthest flow it has. The levels are not strictly
cumulative. A scheme can have Level 3's sideways valuation without Level 2's
in-subtree steering, and in Scrabble the natural path skips Level 2
([below](#why-scrabble-skips-level-2)). Level 4 does contain Level 3.
Steering a probe under DOG by what CAT's probes found means judging an edge
under DOG by probes that never crossed it, which is a global estimate.

**Level 0: no search.** The prior decides alone.

**Level 1: independent averages.**

- A candidate's score is the mean of its own probes' observations.
- Below the root, probes follow a fixed policy and draw from a fixed
  distribution.
- The only adaptivity is at the root: sharing out probes among candidates by
  their own statistics (racing, pruning, successive halving).

So Level 1 estimates each candidate's value *under the rollout policy*: the
outcome if both sides then play as the rollout policy plays. More probes
shrink the noise but never the bias. Suppose the rollout policy never finds
a particular reply. Then no candidate that allows the reply is ever charged
for it, however long the simulation runs.

**Level 2: local search.** Probes steer later probes through the same nodes,
and internal nodes are valued from their own subtrees. As steering below the
root favors better edges, the backed-up root value converges toward the value
under good play rather than under a fixed policy. That removes Level 1's bias.
MCTS with UCT is the standard form, and AlphaZero adds a learned prior. The
information still never leaves the subtree where it was gathered.

**Level 3: sideways valuation.** A probe of one branch prices another. The
exact version shares statistics between two paths to the same state (a
transposition). An approximate version is RAVE, a standard part of the strong
pre-neural Go programs. RAVE values move `a` at node `n` using every probe
below `n` that played `a` at any later point, whether or not it played `a` at
`n`. In Scrabble the approximate kind is what matters, because branches are
rarely identical but often alike. What carries the information is a model of
how outcomes vary across branches, conditioned on the record.

**Level 4: sideways steering.** A probe of one branch changes how probes of
other branches are walked. For example, Bob's reply in DOG's rollouts is the
one CAT's rollouts discovered, and Bob's racks after DOG are drawn more
heavily from the rack region that CAT's rollouts found decisive. Deterministic
search has used the idea for decades: the killer and history heuristics order
the moves at a node by what caused cutoffs in other branches.

**Level 5: a revisable record.** Earlier observations are re-read as the
search learns. Take a probe that was walked with a reply now known to be
inferior. Level 5 re-prices it, or re-runs it where the correction matters,
instead of averaging it in as taken or discarding it. From Level 2 up,
steering improves during the search, so early probes were walked by a worse
policy than late ones. MCTS tolerates this because its visits concentrate
over time. A Scrabble turn affords a few thousand probes, though, and the
early ones are a large share of them. At Level 5 information flows backward
in time.

## BestBot is Level 1

BestBot is the bot code `SIMMING_BOT`, implemented in `ai/bot/elite.go`. Paths
in this section are in the Macondo checkout at `/workspace/mount/macondo`,
tag v0.13.2.

- **Midgame, more than 14 tiles unseen:** it simulates the top 40 moves by
  static equity for 5 plies (the production setting).
- **9 to 14 tiles unseen:** it simulates the top 80, for as many plies as
  there are unseen tiles.
- **8 unseen:** the pre-endgame solver.
- **7 or fewer:** the endgame solver.

The two solvers are outside this document. The pre-endgame solver enumerates
the draws instead of sampling them, and an endgame has no chance nodes left.

How one simulation iteration runs (`montecarlo/montecarlo.go`):

- It draws one opponent rack uniformly from the unseen tiles.
- Every surviving candidate is played against that rack, from the same bag
  state.
- Each candidate is followed by the rollout plies, and every ply is the
  static-equity argmax.
- The leaf observation is a win probability, looked up in a table by tiles
  unseen and by the final spread plus the leave values of the last two
  rollout moves.
- A candidate's score is the mean of its observations. The pick is the
  highest mean, with ties broken by mean equity.

Every 128 iterations, a candidate is pruned when the leader's lower confidence
bound exceeds the candidate's upper bound (`montecarlo/stopping_condition.go`).
The simulation stops when only the leader survives, or at a hard cap.

Its profile:

| | steering | valuation |
|---|---|---|
| root | local: pruning against the leader, each candidate judged by its own probes | local: the mean of its own probes |
| action nodes below the root | non-adaptive: the static-equity argmax | none: nodes below the root are never valued |
| chance nodes | non-adaptive: uniform draws | local: a plain mean within the candidate |

Three features look like more than Level 1 but are not:

- **Shared draws.** Within an iteration, every candidate faces the same
  opponent rack (common random numbers). That couples the probes of different
  candidates, so luck cancels when they are compared. But the draw is fixed
  before either candidate is walked, so nothing learned under CAT reaches DOG.
  The pruning test does not even use the pairing: each candidate's standard
  error comes from its own samples.
- **Pruning** is root steering by the candidates' own statistics, which Level
  1 allows.
- **Rack inference** belongs to a different bot, `SIMMING_INFER_BOT`, not to
  BestBot. It reads the opponent's last move, which is outside the tree. But
  it is computed before the simulation starts and never updated by it, so it
  is a prior.

What BestBot measures, then, is each candidate's win probability when both
players spend the next five plies playing static equity's favorite move. A
setup that pays off only if the opponent fails to block is priced against a
rollout opponent who never blocks deliberately. A reply that static equity
never ranks first is charged to no candidate. More iterations change neither.

## Why Scrabble skips Level 2

Level 2 works by revisiting nodes, and Scrabble's chance nodes make revisits
rare. The following counts use the standard English tile distribution and
random unseen pools of each size:

- **Full racks.** A thousand probes after one candidate land on at least 950
  distinct opponent racks whenever 25 or more tiles are unseen, and on 980 or
  more from 40. From the full bag it is 997 of 1000, out of 3.2 million
  possible racks. Revisits become common only below about 15 unseen tiles,
  which is pre-endgame territory.
- **Face-up leaves.** The opponent draws only the tiles they played, so
  repeats are more common. A three-tile draw repeats in a third to nearly
  half of 1000 probes, and a four-tile draw in a tenth to a sixth.

One ply further down, the mover's own draw multiplies the space again.

Below the first draw, then, most local statistics hold a single probe, and
local steering has nothing to go on. Level 2 there reduces to Level 1. Exact
transpositions run into the same problem.

Yet the branches are alike. Every branch shares the board. What decides a
rollout (a hot lane, a hook, which tiles are still out) is usually a property
of a region of racks and a region of the board, not of one rack. So
information has to move between branches that are similar, not identical, and
what carries it must know what "similar" means. That argues for going
straight to Levels 3 and 4, with a learned model as the carrier.

## Where Scribblez sits

| agent or plan | level | what reaches past Level 1 |
|---|---|---|
| HastyBot, NeuralAgent | 0 | — |
| `sim`, `neural-sim`, `mset-sim` agents | 1 | Nothing. They differ from BestBot in the prior that picks candidates and in their rollouts: to the game's end or to a learned horizon, with every candidate simulated equally. |
| UltimateBot ([sim_residual_feedback.md](plans/sim_residual_feedback.md)) | 3, at the root | Candidates not yet simulated are priced from the simulated ones' evidence, and that price picks the next candidate to simulate. The final pick is still each simulated candidate's own win rate. |
| [rack_conditional_evidence.md](plans/rack_conditional_evidence.md) (proposed) | 5 | Sideways valuation across candidates and racks (Level 3). CAT's discovered replies played in DOG's rollouts (Level 4). Outdated rollouts re-priced (Level 5). |

### What the upper levels need from the model

At Level 1 the model only chooses the candidates, and the rollouts do the
rest. From Level 3 up, the model carries the information between branches: it
says what CAT's probes imply for DOG, or for a rack nobody drew.

- Its error on a branch becomes the error of every estimate that flows
  through that branch.
- Sideways valuation is only as good as the model's grasp of the
  *difference* between siblings.

The self-play corpus shows the model the branches that self-play takes, one
per turn. The model can only extrapolate to a sibling the self-play policy
never plays. [sim_labeled_candidates.md](plans/sim_labeled_candidates.md)
measured a case where that extrapolation fails: the teacher prices a setup
play ten points below its simulated value, because HastyBot's top-10 cut never
plays setups. The plan's simulation rows put K siblings from each sampled
position into the teacher's training, with simulated targets. The models that
carry the flow are distilled from that teacher, and the rows give them two
things the upper levels need:

- coverage of the branches the self-play policy avoids;
- direct supervision on the contrasts between siblings, which is exactly
  what sideways valuation reads.

The labels are Level 1 simulations. They need no model, so they never go
stale, but they carry Level 1's rollout-policy bias. Removing that bias is the
job of the search levels above Level 1, not of the labels.
