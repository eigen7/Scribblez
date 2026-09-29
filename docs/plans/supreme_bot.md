# SupremeBot: a learned search over a probe context

**Status: proposed; not plan-reviewed; nothing built.**

**Goal.** An agent whose search is read and steered by one network that sees
everything the turn's search has done so far. Any probe can then change any
decision anywhere in the tree: the value of an edge it never crossed, a choice
in a branch it shares no ancestor with, or how an earlier probe is read.

**Decision.** The turn's search is a sequence of **probes**, root-to-leaf
paths. Every step of every probe is appended as a token to one **context**.
One causal transformer reads the context to make every decision: the moves
inside probes, the draws inside probes, and the final pick. No statistics are
attached to tree nodes, and no rule says how a node's value is computed from
its children's. Tokens describe what happened (tiles, moves, outcomes), not
where it happened in the tree, so similar branches meet in attention whatever
their tree position. Build it in order: first a learned reader over probes
from a fixed policy, then learned move choices, then learned draws.

## Purpose

[simulation_information_flow.md](../simulation_information_flow.md) names five
ways a probe's information can travel: averaging, local steering, sideways
valuation, sideways steering and revision. Every scheme built or proposed so
far fixes, in code, which of them exist and what carries each.

- BestBot averages.
- MCTS stores statistics at nodes and backs them up from children to parents.
- [rack_conditional_evidence.md](rack_conditional_evidence.md) keeps one
  token per sampled rack, fuses the tokens into each candidate's encoding
  once per block, and lets the learned policy act at ply one only.

Each rule is a guess about where useful information lives, and each one
blocks the flows it did not anticipate. SupremeBot makes no such guess. Two
requirements define it:

1. **A probe of move A at node N can change the preference between moves B
   and C at node M,** even when N and M share no ancestor below the root.
2. **A probe of draw D at node N can change the preference between moves E
   and F at node M,** under the same condition.

The motivating case is transfer between similar racks. Say the opponent's rack
is sampled as QUIZETH, and the opponent plays a high-scoring QUIZ. That says
nearly as much about the sibling sample QUIZATH, which also holds QUIZ. Later,
many probes on both samples have played QUIZ, and then one probe finds that on
QUIZETH, ZIT is better than QUIZ. That finding should transfer to QUIZATH,
which also holds Z, I and T. It should not transfer to a sample without a T.
Nothing stored per node can express this: the two samples are different
chance edges, their subtrees are disjoint, and the finding changes how
probes recorded *before* it should be read.

Where the sibling plans hand-build one of these flows, SupremeBot learns all
five from one mechanism. That is the whole bet, and it is also the risk: the
rules MCTS assumes are what make it converge with no training at all
([Risks](#risks)).

## Where this sits

- **[simulation_information_flow.md](../simulation_information_flow.md)**
  supplies the vocabulary used here. Its idealized agent at Richards–Johnson
  is the behavior SupremeBot aims to learn. In that document's map of schemes,
  SupremeBot's profile is "sideways" in every column, plus revision, which is
  the same row as the idealized agent's. The difference is that every cell is
  learned rather than engineered.
- **[rack_conditional_evidence.md](rack_conditional_evidence.md)** is the
  nearest engineered design. It commits to late fusion so that the context
  is read once per block, not once per rollout, and to a learned policy at ply
  one only. SupremeBot drops both commitments: the context is read at every
  decision, at every ply. Its first layers are prerequisites here too
  ([Build order](#build-order)). Past them, the two designs compete for the
  same slot ([Open questions](#open-questions)).
- **[sim_residual_feedback.md](sim_residual_feedback.md)** supplies
  principles SupremeBot keeps. The model sees its prior's prediction next to
  each observation, and an empty context reduces the model to the plain
  student.
- **[design.md §3](../design.md)** (the public belief system) and
  [roadmap.md's parked rack inference](../roadmap.md#rack-inference-parked)
  are what SupremeBot's learned draws replace once the project leaves
  face-up leaves ([Rack inference is a draw decision](#rack-inference-is-a-draw-decision)).
- **[design.md §8.1](../design.md)** (search-derived knowledge buffers)
  describes the idea in the abstract. In SupremeBot, the buffer is the
  context itself.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)** and
  [blind_spots.md](../blind_spots.md) supply the first training labels
  ([Training](#training)).

## The loop

One turn:

1. The context starts with the **root prefix**: the root position's board
   tokens from the trunk, and one token per root candidate from the move
   proposal model's shortlist, each carrying that model's prior prediction.
2. **Probe.** Walk from the root to a leaf. At each node, ask the network for
   a decision and append the resulting step as a token. At an action node,
   the decision is a move; at a chance node, it is a draw. The leaf is where
   the value-truncated rollout ([roadmap.md item 2](../roadmap.md)) ends: the
   leaf model's reading, or the terminal result.
3. **Repeat** until the budget is spent.
4. **Pick.** Query the network for a value per root candidate, and play the
   best.

Workers run many probes at once. They advance in **ticks**: in each tick,
every probe's pending decision is batched into one network call against the
context as it stood at the start of the tick, and the resulting step tokens are
appended at the end of the tick. Decisions within the same tick do not see
each other. That staleness is the price of batching, and tick size trades it
against throughput. Each tick appends its steps in a fixed order, so a
recorded turn replays exactly: the context the model reads at deployment is the
context training reconstructs.

## Tokens

A probe is a sequence of step tokens. Every token is built from exact
quantities that the engine computes; nothing in a token is a learned
statistic.

| token | contents |
|---|---|
| root board | the trunk's board tokens for the root position, computed once per turn |
| root candidate | the move's footprint, tiles, score and leave; the prior's value prediction for it |
| action step | the mover; the move as footprint cells, tiles, score and leave; its rank and value under the prior among the legal moves at that node; the tiles left in the bag |
| chance step | who drew; the tiles drawn; the resulting rack; the draw's log-probability under the uninformed prior (exact hypergeometric over the unseen pool) and under the distribution it was actually sampled from |
| leaf | the horizon outcome (WLD and score-difference moments, root-mover POV); terminal or truncated; the prior's prediction for the probe's root candidate, so the residual forms inside the model |

Every step also carries its **address**: the probe id, the step's depth, the
root candidate the probe went through, and a relative pointer to its parent
step. The address is a feature, not a route. Attention may use it (to follow
one probe's path, or group probes under one candidate), but nothing forces
information to travel along it.

**Addressing by content is what makes transfer possible.** QUIZETH and
QUIZATH become chance-step tokens that differ in one tile count. The QUIZ
plays that follow them become action-step tokens with the same footprint,
tiles and score. Attention matches them on those features. A scheme keyed by
node identity would see two unrelated edges.

**The board at deep nodes.** A deep node's board is the root board plus the
moves along the probe's path. Tokens carry moves as deltas against the root
board, and the network composes them; there is no trunk encode per node. This
is what keeps a step cheap. It is also the most doubtful representational
choice in the design: the network must infer, from a few move tokens, what a
per-node encode would state outright. The engine can add exact local
features (for example the post-move cross-check delta,
[lexical_features_for_value.md](lexical_features_for_value.md)) without
running the trunk. Whether that suffices is measured at M1 against a variant
that pays for per-node encodes ([Open questions](#open-questions)).

## The network

One causal transformer over the context, with three kinds of **query**.
Queries attend to the context's cached keys and values, but are not
themselves appended to it.

- **Move queries** at an action node. The engine generates the legal moves.
  The prior shortlists them, for example to its top 16, so query count does
  not scale with the full move list. Each shortlisted move becomes a query
  token built like an action step. The output is a logit per move.
- **Draw queries** at a chance node. The output is a proposal distribution
  over draws. It starts as the uninformed prior and stays there until M4
  ([Build order](#build-order)). Draws are sampled from the proposal, and both
  log-probabilities go into the chance-step token.
- **Pick queries**, one per root candidate. The output is the candidate's
  value. The final pick is the argmax.

**Why causal attention.** This was chosen over bidirectional recomputation for
three reasons.

- **Cost.** Each decision reads a KV cache: linear in the context, not
  quadratic.
- **Training efficiency.** A recorded turn is one sequence, and one forward
  pass produces the loss at every decision point and every prefix length, as
  in language-model training ([Training](#training)).
- **Replay.** A token's representation never changes once it is computed, so
  an interrupted or batched search reproduces exactly.

**What causal attention costs.** Revision, the fifth flow, cannot rewrite an
earlier token. When the ZIT finding arrives, the QUIZATH probes keep the
representations they were given. The finding lives in the ZIT probe's tokens.
A later query that reads both the old probes and the new one has to combine
them itself. This works in principle, because every query attends to
everything, but it puts the whole burden of reinterpretation on the reading
side. If that proves too weak, the fix stays inside the causal design:
periodically append a **summary token**, a step with no game content whose
job is to attend to everything so far. Later queries can read the
reinterpretation there instead of recomputing it ([Open
questions](#open-questions)).

**The floor.** With an empty context, a pick query must return the prior's
prediction, and a move query must return the plain student's policy. Training
includes empty and near-empty prefixes. So the untrained end of the budget
curve is the current one-pass agent, never worse.

## Probes are experiments, not samples

In averaging schemes, a probe must be a faithful sample of the game: the
opponent must play what they would really play, and the draws must follow the
bag's probabilities, or the average is biased. SupremeBot's reader is learned,
and it sees every choice that produced a probe: each move's rank under the
prior, and each draw's true and proposal log-probabilities. It is trained to
value candidates correctly, whatever it was shown. So a probe can be an
experiment:

- An opponent can try a setup play no static policy would choose, to find out
  whether it wins. Probe 3 of the Richards walkthrough is this.
- A draw can be forced into the rack region that decides a candidate. Probe 2
  of the walkthrough is this.

Correcting for these choices is not a separate importance-weighting step.
The reader learns it, because its labels are true values and its inputs state
how each probe was chosen. The correction holds only for probe policies the
reader has been trained against, so the reader and the probe policy are
trained together ([Training](#training)).

## Rack inference is a draw decision

Under face-up leaves, the only hidden tiles are fresh draws, and the
hypergeometric prior over the unseen pool is the true distribution. In
standard Scrabble, the opponent's kept tiles are hidden too, and the true
distribution is a posterior: the opponent's past plays, exchanges and passes
say which racks they probably hold. Existing engines compute that posterior in
a separate module and sample from it. Macondo's `SIMMING_INFER_BOT` does this,
and so does the port in
[belief/rack_inference.h](../../engine/include/belief/rack_inference.h).
SupremeBot needs no such module, because inference falls to two parts it
already has.

- **The reader weights.** Chance steps carry the uninformed prior's
  log-probability, not the posterior's. The pick labels are values against
  the rack the opponent actually held in the recorded game, so a reader
  trained on them values candidates under the posterior the training games
  actually produced. It learns to discount probes on racks the opponent's
  history rules out, with no likelihood model anywhere. The root prefix
  carries that history: the opponent's past moves as tokens.
- **The writer samples.** A reader that only reweights is doing importance
  sampling from the uninformed prior. That wastes most probes when the
  posterior is sharp, for example after a play that tells which five tiles the
  opponent kept. A learned draw proposal puts probes where the posterior
  mass is. It is trained by the same telescoping reward as the move choices,
  so it learns what to sample, not a likelihood. That can differ from the
  posterior: the proposal should favor racks where candidates disagree, and
  avoid racks where the posterior is high but every candidate does the same.
  A separate inference module cannot make that trade, because it does not
  know what the search is trying to decide.

The costs of this approach:

- **The posterior is the one the training opponents produce.** Inference is
  only as good as the match between training opponents and real ones. A
  separate module has the same dependence through its likelihood model, but
  there it is a stated parameter, the likelihood temperature. Here it is
  implicit in the training data, which is an argument for opponent diversity
  in the corpus, and possibly for an opponent-identity token.
- **The labels are noisy.** A value against the one rack the opponent held is
  one sample from the posterior. It is unbiased, and it needs many positions.
- **There is a baseline to beat.** The ported inference makes a comparison
  arm, with draws sampled from its posterior and a reader trained over them.
  Its log-probability can also go into the chance-step token as a hint the
  network is free to ignore.

## Training

The network plays two roles:

- The **reader** answers pick queries.
- The **writer** answers move and draw queries.

The reader is trained with supervised learning, and first. The writer is
trained on the reader's improvement, and second.

### The reader

A training row is one recorded turn: the context, plus a **label** for each
root candidate, meaning its value under a reference search much stronger than
the budget being trained.

- Pick queries are applied after every leaf token, so one sequence supervises
  the reader at every budget from one probe to the full record. The loss is
  value regression plus a pairwise ranking term, and the regret of the argmax
  is the headline metric.
- **Label source, first:** large-budget averaging simulations over every
  shortlisted candidate, the target stream of
  [sim_labeled_candidates.md](sim_labeled_candidates.md). These labels carry
  their rollout policy's bias: they never learn of a YEET that hasty would not
  play ([simulation_information_flow.md](../simulation_information_flow.md#what-the-sideways-flows-need-from-the-model)).
  So they can teach the reader to match a large-budget averager at a fraction
  of its budget, but not to beat it.
- **Label source, then:** SupremeBot labels itself. A run at many times the
  training budget labels the positions for runs at the training budget, as in
  expert iteration: search distills into less search. This is where
  SupremeBot can exceed its first teacher. It starts only once the writer is
  learned, because a fixed writer plus reader is bounded by what the fixed
  writer's probes can reveal.
- **Subset assembly** is valid while the writer is fixed: probes are then
  independent given the root, so any subset of a turn's probes, in any order,
  is a context the deployed agent could have produced. This multiplies rows
  cheaply. Once the writer reads the context, probes depend on the earlier
  ones, and rows must be the recorded sequences as they were produced. This is
  the same constraint as rack_conditional_evidence.md's "targets from a
  context-conditioned policy".

### The writer

- **Start by imitation.** Move queries imitate the plain student's policy;
  draw queries output the true distribution. With that writer, SupremeBot is
  a learned reader over ordinary rollouts, which is M1.
- **Then reinforcement learning.** The writer's purpose is to make the pick
  better. Define the potential of a prefix as the label value of the reader's
  current argmax, and a probe's reward as the change in potential across it.
  The rewards telescope: summed over a turn, they equal the final pick's label
  value minus the prior's pick's. Each probe gets credit for the improvement
  it caused, and the objective is exactly the final pick's quality. This needs
  a label for every candidate the reader may pick, so during training the pick
  is restricted to labeled candidates.
- **Alternate the two roles.** A new writer changes the reader's input
  distribution, so the reader retrains on the new writer's turns before the
  writer steps again, in the manner of AlphaZero's generations
  ([generational_training.md](../generational_training.md)).

### Budget generalization

A learned search is only as good as the budgets it was trained at. Rows span
budgets from zero to beyond the deployment budget, and the headline curve is
regret against budget. It must keep falling past the largest training budget.
A curve that flattens there means SupremeBot has learned a budget-specific
routine rather than how to search.

## Cost

Reading the context is cheap. Here is an order-of-magnitude estimate under
stated assumptions, to be replaced by a measurement at M1:

- 2,000 probes of about four plies give about 8,000 action decisions and a
  context of about 20,000 tokens (action, chance and leaf steps).
- Each decision scores a shortlist of 16 moves against a cache averaging
  10,000 tokens: 8,000 × 16 × 10,000 ≈ 1.3 × 10⁹ query–key pairs per layer
  per turn.
- At model width 128, counting both the score and the value products, that
  is about 6.6 × 10¹¹ FLOPs per layer, and about 4 × 10¹² for six layers.
- A 4090 peaks around 1.65 × 10¹⁴ bf16 FLOP/s, so the ideal time is about
  25 ms per turn. The achieved time is several times that, and still small
  against the rollouts.
- The KV cache for 20,000 tokens is about 60 MB at six layers, width 128, in
  bf16.

So rack_conditional_evidence.md's argument for late fusion, that the context
must not be read per rollout, does not bind at these sizes once a read is
linear. The costs that do bind lie outside attention:

- **Full move generation at every queried node.** Greedy hasty stops its
  search without listing the legal moves, and full generation costs about
  twice as much (rack_conditional_evidence.md, reader 2). SupremeBot pays this
  at every node it queries, not only at ply one.
- **A GPU round trip in the middle of every probe.** Ticks turn this into one
  batched call per ply per tick. Probes stall at each call, so throughput
  depends on keeping enough probes in flight.
- **The board at deep nodes.** Move deltas cost nothing extra. A trunk encode
  per node would cost one trunk pass per decision, 8,000 per turn above, and
  would dominate everything else.

The lever for all three is the same: **query scope**. Nodes outside the scope
fall back to the prior's argmax with hasty's cheap generation. Start with ply
one and the probe's first own move, and widen the scope when a measurement
says it pays.

## Risks

**No free convergence.** MCTS improves with more search even with a bad
network, because its backup rules are sound. SupremeBot has no such rules. If
the reader has learned nothing useful about a region of positions, more probes
do not help there. The floor ([The network](#the-network)) guarantees the
prior's quality, not improvement over it. The budget curve is the guard.

**Transfer can go wrong in both directions.** The reader may fail to transfer
what should transfer (ZIT to QUIZATH), or transfer what should not (ZIT to a
rack without a T). Both are measured directly at M2 with synthetic contexts
built to contain exactly one such fact.

**Throughput.** If full move generation, the round trips and the deep-node
boards hold probes per second far below hasty rollouts even at ply-one scope,
the learned writer is not affordable. SupremeBot then reduces to M1's learned
reader over cheap probes. That is still a scheme with every sideways valuation
flow, and it reaches rack_conditional_evidence.md's destination by a different
route.

**Opacity.** When a known case fails, there is no node table to inspect. A
bug looks exactly like "the model did not learn". Each milestone therefore
comes with its own diagnostic: identical-record comparisons at M1, synthetic
single-fact contexts at M2, and attention attribution from the pick query back
to the probes that moved it.

**The labels' bias.** Until self-labeling starts, labels inherit hasty's
blind spots, and the reader learns to reproduce them faster, not to remove
them.

## Build order

Each milestone produces a working agent, measured before the next begins.

- **M0: the record.** Per-step probe logging (tiles, moves, outcomes, the
  prior's ranks and both draw probabilities), extending
  rack_conditional_evidence.md's layer 1 from per-rollout to per-step. Plus
  the token encoder, and turns replayable from the record.
- **M1: learned reader, fixed writer.** The writer is the plain student at
  ply one, then hasty, with true draws. It is measured on **identical
  records**: the reader and plain averaging value the same probes, so the
  comparison isolates the valuation. The report is regret against budget.
  *Kill criterion:* if the reader does not beat averaging on identical records
  at matched budgets, stop. If it passes, match play against UltimateBot and
  BestBot. This milestone also measures deep-node boards as move deltas
  against per-node encodes.
- **M2: transfer tests.** Synthetic contexts, each built to contain exactly
  one fact, plus its control: QUIZETH to QUIZATH, and the no-T control. Then
  the Richards–Johnson position and the ACETA family in
  `positions/NWL23/interesting-positions/`.
- **M3: learned move choices.** The writer is trained with the telescoping
  reward, alternating with the reader. Measured in match play against M1 at
  equal wall-clock time, not equal probes, because steering costs time.
- **M4: learned draws.** Proposal distributions at chance nodes. Under
  face-up leaves this comes last, because it distorts the reader's input
  distribution the most and matters least there. It is also the gateway to
  standard Scrabble, where it becomes the rack inference
  ([Rack inference is a draw decision](#rack-inference-is-a-draw-decision)),
  measured against the ported posterior.
- **M5: self-labeling.** SupremeBot at many times the budget labels
  SupremeBot's training positions.

## Open questions

- **SupremeBot and rack_conditional_evidence.md.** Their first layers are
  shared, namely per-rollout logging and the plain student at ply one. After
  that, both designs claim the next build slot. One option: M1 is built in
  place of that plan's layer 4, with its aggregate model as M1's baseline.
- **Deep-node boards:** move deltas over the root board, or a trunk encode per
  node. M1 measures both.
- **Summary tokens:** whether causal revision needs them, and if so, their
  schedule and what trains them.
- **Stopping:** a fixed budget first. A learned stop head fits the same
  query mechanism and the telescoping reward, less a cost per probe.
- **Tick size:** the batching staleness against throughput.
- **Context across turns:** discarded at the end of each turn, as in every
  scheme so far. Carrying over the probes that remain legal is deferred until a
  case needs it.
- **Standard Scrabble:** under face-up leaves, a chance step draws the
  opponent's replenishment tiles only. Nothing in the tokens assumes that. The
  hidden full rack is one more chance step, and the opponent's history goes
  in the root prefix, so the move is a corpus regeneration, not a redesign.
  What is open is whether the reader's implicit posterior is good enough
  before M4, or whether standard Scrabble has to wait for learned draws.
