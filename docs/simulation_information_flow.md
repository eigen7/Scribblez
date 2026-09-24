# Information flow in simulation search

**Purpose.** Agents that decide by simulation differ most in one respect: how
far what one simulation learns travels through the game tree. This document
does three things:

- It follows an idealized agent through a famous position, to show what that
  travel buys.
- It gives a vocabulary for classifying schemes by it.
- It places Macondo's BestBot, the reference opponent this project measures
  against ([evaluation_plan.md](evaluation_plan.md)), as the simplest scheme
  that simulates at all. In BestBot, a simulation informs one number: the
  average of the candidate it started from.

The schemes beyond BestBot are where Scribblez's search work points
([sim_residual_feedback.md](plans/sim_residual_feedback.md),
[rack_conditional_evidence.md](plans/rack_conditional_evidence.md)). They lean
on a model that [sim_labeled_candidates.md](plans/sim_labeled_candidates.md)
sets out to fix.

## An idealized agent at Richards–Johnson

<img align="right" width="45%" alt="The critical position at move 22" src="analysis/richards-johnson-exchange/images/critical-position.png">

The position is Nigel Richards's EELLT exchange from the 2025 World Cup,
analyzed in [analysis/richards-johnson-exchange/](analysis/richards-johnson-exchange/README.md):

- Nigel leads Mike Johnson 379 to 326 and holds AEELLNT.
- Seven tiles are in the bag, and fourteen are unseen: A C D E E I I L N R S T
  T U.
- The engines' favorite is B6 ALLEE for 16.
- Nigel exchanged EELLT and kept AN, a play no engine setting approves of.

The analysis argues that the exchange wins, on timing.

Below, an idealized simulating agent reaches Nigel's conclusion. It runs one
thread; a real agent would walk many paths at once. Each **probe** walks from
the position to the end of the game, choosing Nigel's and Mike's moves and
every draw, and records what happened. The agent makes all of those choices
adaptively, and it reads each probe for more than its score.

The agent starts with a **prior** that already holds the reads the analysis
makes before weighing any play:

- WOE points to a bingo-prone rack for Mike, with an S.
- ZONES and WOE together point to the C being in the bag.

These reads shape where the first probes go, but simulation did not teach
them.

<br clear="all"/>

**Probe 1: ALLEE, and a bingo.** The agent starts with the static favorite
and draws Mike a rack from the prior. Nigel plays ALLEE and draws. Mike plays
14E RULIEST for 70 and leads by one. Nigel holds CEINNTT: he has the C, but
no CLOSE hook near a triple, and only A and D are unseen. Mike goes out with
them and wins ([the analysis's ALLEE line](analysis/richards-johnson-exchange/README.md#why-not-allee--the-worst-case)).

A plain simulator records a loss. The idealized agent records why. ALLEE left
two tiles in the bag, so the bingo left Mike two tiles from going out, and
Nigel had a single turn to answer. The explanation never mentions RULIEST: it
holds for every rack Mike can bingo with.

**Probe 2: draw where the losses are.** The losses live on Mike's bingo racks,
so the next ALLEE probes draw his racks mostly from there. When the agent
values ALLEE, it weights those racks back down to their true probability.
After the bingo, the agent searches Nigel's reply rather than taking one fixed
policy's move. It asks whether the draws that bring the C also bring the
triple-word hook. They sometimes do. But the C is only about 5/7 after ALLEE
even if it is in the bag, and holding it does not guarantee the hook.

**Probe 3: a reply no static policy plays.** With E, T and S in his rack, Mike
need not bingo at once. The agent tries 8K YEET for 7, a play static equity
would never choose, keeping the S. Now an S out-bingo has two lanes: WOES at
J14, and YEETS at O8, a lane worth nearly 100. Nigel can block only one
([the double-S danger](analysis/richards-johnson-exchange/README.md#the-ye_-and-double-s-danger)).

That path used one seven-tile rack for Mike. Treating his rack as a uniform
draw from the fourteen unseen tiles (before any inference):

- He has 1,346 distinct possible racks, and none is more likely than 0.23%.
- The lesson of the probe covers every rack with an E, a T and an S. That is
  246 racks and 26% of the probability, before the WOE read raises the S.

So one path, a vanishing sliver of the rack space, generalizes to a quarter of
it. In the game, Mike held DEIRSTU.

**Probe 4: what the lessons say about moves not yet simulated.** Neither
mechanism is specific to ALLEE. Any scoring play that leaves two tiles in the
bag walks into the same timing trap, and the YE_ setups ride on that timing.
So the agent does two things without walking a single new probe:

- It marks down every such candidate.
- In later probes of those candidates, Mike tries the YE_ setup whenever his
  rack allows it.

The mechanisms also say what an answer must do: leave the bag full at Mike's
turn, so that a bingo leaves him seven tiles and no way out. An exchange does
that, and still draws Nigel five fresh tiles to chase the C. Static equity
ranks exchanges far down its list. The agent simulates one next because of
what the ALLEE probes found.

**Probe 5: the exchange.** Nigel throws back EELLT and keeps AN. Mike bingos
through WOES for about 70 and leads by about 20. But seven tiles are still in
the bag, so he draws a full rack and cannot go out. Nigel has drawn the C, and
with AN kept he likely has a four-letter C word through the triple word at
H15 (CANE, CAIN, CLAN, CARN, CANS, CANT; the game's H12 CLAN scored 37). He
scores there, then goes out.

Probe 5's lessons rank the other exchanges before any of them is simulated:

- Whatever Nigel throws back lands on Mike's rack after his bingo, so the
  tiles that duplicate the unseen E's and T's cramp Mike most.
- Of the leaves Nigel could keep, AN makes the C most useful: it lets 9 of the
  13 other unseen tiles complete a four-letter C word, against 5 for EN.

**Probe 6: revisiting old probes.** The first ALLEE probes were walked before
the agent knew about YEET. In the ones where Mike held E, T and S and played
something else, he played worse than he could have. The agent re-prices those
probes, or re-runs the ones where it matters, instead of letting them flatter
ALLEE's average.

The agent ends where Nigel did.

### What that took

Each step moved information somewhere a plain simulator does not let it go:

| step | what moved | the flow ([defined below](#flows)) |
|---|---|---|
| 1 | a probe recorded a mechanism, not just an outcome | the precondition for every other flow |
| 2 | ALLEE's probes chose the racks and replies of later ALLEE probes | local steering, at chance and action nodes |
| 3 | one rack's result priced a quarter of the rack space | sideways valuation, at a chance node |
| 4 | ALLEE's probes priced other candidates, chose the next candidate, and changed Mike's replies in other candidates' probes | sideways valuation and sideways steering, at the root and below it |
| 5 | one exchange's result ranked its sibling exchanges | sideways valuation, at the root |
| 6 | later probes corrected earlier ones | revision |

BestBot takes none of these steps. Beyond averaging, its only use of the
record is to stop simulating a candidate that falls behind
([below](#bestbot-the-simplest-scheme)).

## The vocabulary

### Simulation schemes

A simulating agent searches a game tree rooted at the current position. The
tree has two kinds of node:

- **Action nodes**, where a player chooses a move. The root is one, and its
  edges are the candidate moves.
- **Chance nodes**, where the game deals. Examples are bag draws, and the
  opponent's unknown rack (under face-up leaves, only the tiles the opponent
  drew are unknown).

Every such agent runs the same loop:

1. **Probe.** Walk from the root to a leaf, choosing an edge at every node.
2. **Record.** Read an observation at the leaf and add it, with the path, to
   the **record**. The observation may be an outcome, an evaluation of the
   horizon position, or something richer: which replies were played, on which
   racks, and why the game went as it did.
3. **Repeat** until the budget is spent or a stopping rule fires.
4. **Decide.** Score every root candidate from the record, and play the best.

Two things read the record:

- **Steering** is how the record shapes the next probe, that is, which edge
  the probe takes at each node.
- **Valuation** is how the record prices an edge.

Everything known before the first probe is the **prior**: static equity, leave
values, a trained network, rack inference from the opponent's last plays. A
prior can inform every step, but it carries nothing from one probe to another.
The classification concerns the record only.

### Locality is decided at the edge

Write `S(e)` for the probes that crossed edge `e`. An estimate attached to `e`
is **local** if it reads only `S(e)`, and **global** if it also reads probes
that went elsewhere.

The natural first attempt draws the line at the node: a decision at `n` is
local if it reads only the probes that passed through `n`. At the root every
probe passes through, so every root decision would count as local. The
question that matters most, whether ALLEE's probes say anything about the
exchange, would then have no answer. Drawn at the edge, the line answers it.

- **Valuation** is local or global according to the estimate it produces.
- **Steering** is **fixed** when it ignores the record. A static rollout
  policy and draws from the prior's distribution are both fixed.
- Otherwise steering has the scope of the edge estimates it compares.
  Pruning a candidate against the leader is local steering: each candidate is
  judged by its own probes, and comparing candidates is what steering does.

### Action nodes and chance nodes

Scope applies at both kinds of node, and each combination asks a different
question. In the Richards position:

| | valuation | steering |
|---|---|---|
| action node | Do ALLEE's probes tell us what the exchange is worth? | Does a probe where Mike bingos after ALLEE decide which candidate we simulate next, or which reply Mike tries after the exchange? |
| chance node | Does Mike's result on one ETS rack tell us his result on another? | Do the racks that decided ALLEE change which racks we draw for Mike after the exchange? |

Chance nodes differ in one respect. An action node is worth its best edge, so
steering there should favor good edges. A chance node is worth the
expectation over its draws at their true probabilities. Steering may favor
the informative draws, as probe 2 did, but valuation must then undo the skew
with importance weights. Otherwise the scheme values the draws it chose to
look at, not the ones the bag will deal.

### Flows

A probe's information can move in five ways. They are named by where the
information ends up.

| flow | where a probe's information ends up | examples |
|---|---|---|
| **averaging** | in the value of its own path's edges | every simulator |
| **local steering** | in the choices of later probes through the same nodes | UCT's bandit at each node; AlphaZero |
| **sideways valuation** | in the value of edges it never crossed | a transposition's shared statistics; RAVE in Go, which values a move at a node from every probe below it that played the move later |
| **sideways steering** | in the choices of probes elsewhere in the tree | killer and history move ordering in chess, where what cut off one branch is tried first in others |
| **revision** | in the observations of probes recorded before it | the re-pricing in [rack_conditional_evidence.md](plans/rack_conditional_evidence.md) |

The flows are not a ladder, and they combine freely. A scheme can have
sideways valuation without local steering, and in Scrabble that is the
natural choice ([below](#why-local-steering-is-not-enough-in-scrabble)). There
is one dependency. Sideways steering needs sideways valuation somewhere: to
steer Mike's reply after the exchange by what ALLEE's probes found is to value
an edge below the exchange by probes that never crossed it.

A scheme's **profile** records, for each place (the root, action nodes below
it, chance nodes), which flows reach it.

## A map of schemes

| scheme | root steering | steering below root | chance steering | action valuation | chance valuation | revision |
|---|---|---|---|---|---|---|
| Macondo BestBot | local (pruning) | fixed | fixed | local | local | no |
| Scribblez `sim`, `neural-sim`, `mset-sim` | fixed (top K, simulated equally) | fixed | fixed | local | local | no |
| MCTS with UCT; AlphaZero | local | local | fixed | local | local | no |
| MCTS with RAVE | sideways | sideways | fixed | sideways | local | no |
| UltimateBot ([sim_residual_feedback.md](plans/sim_residual_feedback.md)) | sideways | fixed | fixed | sideways, for steering only | local | no |
| [rack_conditional_evidence.md](plans/rack_conditional_evidence.md) (proposed) | sideways | sideways, at ply one | sideways | sideways | sideways | yes |
| the idealized agent above | sideways | sideways | sideways | sideways | sideways | yes |

UltimateBot's sideways valuation prices unsimulated candidates from the
simulated ones' evidence, and the price picks the next candidate. Its final
pick is still each simulated candidate's own win rate.

## BestBot: the simplest scheme

BestBot is the bot code `SIMMING_BOT`, implemented in `ai/bot/elite.go`. Paths
in this section are in the Macondo checkout at `/workspace/mount/macondo`,
tag v0.13.2.

- **More than 14 tiles unseen:** it simulates the top 40 moves by static
  equity for 5 plies (the production setting).
- **9 to 14 unseen, as in the Richards position:** the top 80, for as many
  plies as there are unseen tiles, which reaches the end of the game in the
  usual case.
- **8 unseen:** the pre-endgame solver, which enumerates the draws rather
  than sampling them.
- **7 or fewer:** the endgame solver, where no chance nodes are left.

How one simulation iteration runs (`montecarlo/montecarlo.go`):

- It draws one opponent rack uniformly from the unseen tiles.
- Every surviving candidate is played against that rack, from the same bag
  state.
- Each candidate is followed by the rollout plies, and every ply is the
  static-equity argmax.
- The leaf observation is a win probability: the outcome if the game ended,
  otherwise a table lookup by tiles unseen and the spread plus leftover leave
  values.
- A candidate's score is the mean of its observations. The pick is the
  highest mean, with ties broken by mean equity.

Every 128 iterations, a candidate is pruned when the leader's lower confidence
bound exceeds the candidate's upper bound (`montecarlo/stopping_condition.go`).

Its profile is the minimum in every cell except root steering, and three
features look like more than they are:

- **Shared draws.** In an iteration every candidate faces the same opponent
  rack (common random numbers), so luck cancels in comparisons. The draw is
  fixed before any candidate is walked, though, so nothing learned under
  ALLEE reaches the exchange. The pruning test does not even use the pairing:
  each candidate's standard error comes from its own samples.
- **Pruning** is root steering by the candidates' own statistics.
- **Rack inference** belongs to a different bot, `SIMMING_INFER_BOT`. It reads
  Mike's last plays, which are outside the tree, but it is computed before the
  simulation starts and never updated: a prior.

Held against the walkthrough:

- BestBot records a win probability and nothing else, so no probe can say why
  it lost.
- It draws Mike's racks uniformly throughout. The WOE read is not in its
  prior.
- Its Mike plays the static-equity argmax, so he never tries YEET. No
  candidate is charged for the double-S danger, however many iterations run.
- It prices each of its 80 candidates only by that candidate's own probes.

The analysis reports that Macondo ranks the exchange anywhere from 12th to
40th.

## Why local steering is not enough in Scrabble

Local steering learns from revisiting nodes, and Scrabble's chance nodes make
revisits rare. In the Richards position, Mike's rack alone has 1,346 possible
values, and even the likeliest comes up only about once in 430 probes. That is a
small space: fourteen tiles unseen, one draw deep. Generally, counting from
the standard English tile distribution over random unseen pools of each size:

- **Full racks.** A thousand probes after one candidate land on at least 950
  distinct opponent racks whenever 25 or more tiles are unseen, and on 980 or
  more from 40. From the full bag it is 997 of 1000, out of 3.2 million
  possible racks.
- **Face-up leaves.** The opponent draws only the tiles they played, so
  repeats are more common. A three-tile draw repeats in a third to nearly
  half of 1000 probes, a four-tile draw in a tenth to a sixth.

One ply further down, the mover's own draw multiplies the space again.

Below the first draw, then, most nodes are visited by a single probe, and
local steering has nothing to work with. Exact transpositions meet the same
problem.

Yet the branches are alike: every branch shares the board. What decides a
rollout is usually a property of a region of racks and a region of the board,
not of one rack. Examples are a hot lane, an S hook, or two tiles left in the
bag, and probe 3's ETS region is the pattern. Information must move between
branches that are similar, not identical, and what carries it has to know what
"similar" means. That is the case for going straight to the sideways flows,
with a learned model as the carrier.

## What the sideways flows need from the model

When a scheme only averages, the model's job is to choose the candidates, and
the rollouts do the rest. Once information flows sideways, the model carries
it between branches. It says what ALLEE's probes imply for the exchange, or
what one ETS rack implies for a rack nobody drew.

- Its error on a branch becomes the error of every estimate that flows
  through that branch.
- Sideways valuation is only as good as the model's grasp of the *difference*
  between siblings.

The self-play corpus shows the model the branches self-play takes, one per
turn. The model can only extrapolate to a sibling the self-play policy never
plays. [sim_labeled_candidates.md](plans/sim_labeled_candidates.md) measured a
case where that extrapolation fails: the teacher prices a setup play ten
points below its simulated value, because HastyBot's top-10 cut never plays
setups. The plan's simulation rows put K siblings from each sampled position
into the teacher's training, with simulated targets. The models that carry the
flow are distilled from that teacher, and the rows give them two things
sideways flow needs:

- coverage of the branches the self-play policy avoids;
- direct supervision on the contrasts between siblings.

The labels come from plain averaging simulations with a fixed rollout policy.
They need no model, so they never go stale. But they carry that policy's
bias: a label never learns of a YEET the rollout policy would not play.
Removing that bias is the search's job, not the labels'.
