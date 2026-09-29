# SupremeBot: a learned search over a probe context

**Status: the project's direction as of 2026-09-29; plan-reviewed the same day
([review record](#review-record)); nothing built.** Development leaves face-up leaves for standard Scrabble, and
the other roadmap items are on hiatus ([roadmap.md](../roadmap.md)).

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
  nearest engineered design, now on hiatus. It commits to late fusion so
  that the context is read once per block, not once per rollout, and to a
  learned policy at ply one only. SupremeBot drops both commitments: the
  context is read at every decision, at every ply. M0 builds its per-rollout
  logging (layer 1, never built), generalized to per-step
  ([Build order](#build-order)).
- **[sim_residual_feedback.md](sim_residual_feedback.md)** supplies
  principles SupremeBot keeps. The model sees its prior's prediction next to
  each observation, and an empty context reduces the model to the plain
  student.
- **[design.md §3](../design.md)** (the public belief system) and
  [roadmap.md's rack inference](../roadmap.md#rack-inference)
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
   tokens from the trunk, one token per root candidate from the move proposal
   model's shortlist, each carrying that model's prior prediction, and the
   opponent-history tokens. The board tokens, candidate encodings and prior
   predictions are what the item-3 cache graph already computes once per turn
   (`MoveProposalService`,
   [move_proposal_service.h](../../engine/include/agent/move_proposal_service.h)),
   so the root prefix reuses that runtime. Its evidence-fusion step graph is
   not used.
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
against throughput.

A tick does not wait for slow probes. A probe whose next step is CPU-bound, an
endgame solve or a long move generation, leaves the tick loop, and its tokens
are appended at whichever later tick it finishes by. So tick membership
depends on timing, and the recorded turn is the ground truth. Every token
records its **tick id**, and replay replays the recorded choices and tick
boundaries. It never recomputes decisions, which batched BF16 inference would
not reproduce bitwise anyway.

## Tokens

A probe is a sequence of step tokens. Every token is built from exact
quantities that the engine computes; nothing in a token is a learned
statistic.

| token | contents |
|---|---|
| root board | the trunk's board tokens for the root position, computed once per turn |
| root candidate | the move's footprint, tiles, score and leave; the prior's value prediction for it |
| opponent history | one per past opponent turn: the move's footprint, tiles and score, and its static-equity rank among the legal moves on the board it was played from (computed by the engine when the move was played); exchanges as a tile count, passes as a flag |
| action step | the mover; the move as footprint cells, tiles, score and leave; the tiles left in the bag; a rank and value, whose source depends on the node ([below](#what-an-action-step-costs)) |
| option | one move recorded at a board ([Move lists](#move-lists-local-and-global)): footprint, tiles and blank designations, tile count, score. Nothing that depends on a rack |
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

### Move lists: local and global

The network has no lexical ability, so the only way it learns what a rack
could play is from the engine's move list. Without one, the ZIT finding cannot
transfer to QUIZATH: the reader has to know that ZIT is playable there, and
in general ("can this rack bingo in that lane?") that is a lexical fact.

A node's full legal list is too large to record. It runs to hundreds of moves,
and to thousands with blanks, which across 2,000 probes would put millions of
tokens in the context. So the list is used in two ways:

| | what it sees | stored in the record |
|---|---|---|
| the decision at the node | the full legal list, exchanges and pass included | no: the move generator is deterministic given board and rack, so replay regenerates it |
| the global context | a recorded subset of **option** tokens, plus the move played | yes |

Full generation and scoring every legal move at a node is not new: UltimateBot
does it at the root today (`equity_top_k` with no limit, then
`encode_candidates`, [ultimate_bot_agent.cpp](../../engine/src/agent/ultimate_bot_agent.cpp)).

The played move always enters the context as its action step, so the record
covers everything any probe actually played. The subset only has to cover the
alternatives worth reasoning about globally. That is its limit: the reader can
reason only about alternatives that were recorded or played, so choosing the
subset is a design question in its own right.

**The selection rule** is static equity's top k plus a few slots reserved for
plays in other board regions: a top k by value alone would drop the setup
plays the design exists to find. Static equity is available at every node
without a trunk pass, so the rule is the same at M1, where the writer is hasty
and there is no prior at ply one, as at M3a. Later, what to record can become a
writer decision, trained by the same reward as the others ([Open
questions](#open-questions)). Exchanges and passes are never recorded as
options: they need no lexical knowledge, since an exchange is a keep-set of
the rack, legal when at least seven tiles are in the bag. An exchange still
becomes a token as a root candidate, so a pick query can value it (Richards's
EELLT), and as an action step when a probe plays one.

**Options are rack-free and recorded once per board.** Every probe through
root candidate X sees the same board after X; only the opponent's rack
differs. An option token carries only what the board determines: placement,
tiles, tile count and score. Leave and equity depend on the rack, so they are
not in the token; the reader derives them from the move's tiles and the rack
it is asking about, as the student already does for leaves
([move_set_encoder.h](../../engine/include/training/move_set_encoder.h)). So at
a shared board each distinct move is recorded once, the first time any rack's
selection includes it.

Whether a rack can play an option is multiset containment, exact and free of
lexicon. With racks encoded as cumulative tile counts ("at least two E's"),
as the unseen-pool thermometer already is, the dot product of the move's
requirement with the rack equals the move's tile count exactly when the rack
contains the move. That is one dot product and one comparison against the tile
count the token carries, which an attention head can compute. Blanks relax it:
the rack plays the move when the shortfall is at most its blank count. So the
reader can tell whether ZIT is playable on QUIZATH before any probe draws
QUIZATH. Past the opponent's reply the boards differ per probe, and each node
records its own options.

**Rows and replay.** An option belongs to its board, but the record also
lists which probes selected it. When subset assembly drops probes, the row
builder recomputes the option set from the probes it keeps, and an option's
tick becomes that of its earliest kept selector. Probes in one tick that
select the same move produce one option, in recorded order.

**Context size.** Rough, at 2,000 probes and 3–4 action nodes per probe; the
typical list length and how fast each shared board's recorded set stops
growing are measured before M0:

| options recorded | extra tokens | context |
|---|---|---|
| full lists at every node | millions | infeasible |
| top 16 at every node | ~128,000 | ~150,000 |
| top 16 at in-scope nodes (ply one, first own move) | ~64,000 | ~85,000 |
| the same, ply-one options once per shared board | ~40,000 | ~60,000 |
| M1: ply-one options only, once per shared board | the saturated set per board, times 16 boards | measured before M0 |

### What an action step costs

The prior's rank and value at a node need a trunk pass: the student's trunk
reads the mover's rack and the pool that rack implies
([game_state_encoder.cpp](../../engine/src/encoding/game_state_encoder.cpp)).
They also need full move generation, which greedy hasty skips
([hasty_bot.h](../../engine/include/agent/hasty_bot.h)). So the fields depend
on the node:

- **Inside the query scope** ([Cost](#cost)), the node pays for a trunk pass
  and full generation, and the step carries the prior's rank and value.
- **Outside it**, the step carries hasty's static-equity rank and score,
  which cheap generation provides.

At ply one this means one trunk pass per probe, because the opponent's rack
differs in every probe. A rack-late student
([rack_conditional_evidence.md](rack_conditional_evidence.md), layer 3) would
remove that cost; it is an option, not a prerequisite.

### The board at deep nodes

A deep node's board is the root board plus the
moves along the probe's path. Tokens carry moves as deltas against the root
board, and the network composes them; there is no trunk encode per node. This
is what keeps a step cheap. It is also the most doubtful representational
choice in the design: the network must infer, from a few move tokens, what a
per-node encode would state outright. The engine can add exact local
features (for example the post-move cross-check delta,
[lexical_features_for_value.md](lexical_features_for_value.md)) without
running the trunk. M1 uses move deltas only. A per-node-encode variant is
built only if M1's regret curve looks limited by the representation
([Open questions](#open-questions)).

## The network

One causal transformer over the context, with three kinds of **query**.
Queries attend to the context's cached keys and values, but are not
themselves appended to it.

- **Move queries** at an action node, a two-level readout over the node's
  full legal list. First a few node-summary queries (say 8) read the context,
  together with the node's trunk board tokens as a side input where the node
  is in scope. Then every legal move, as engine features only, is scored
  against those summaries with a cheap cross-attention, the shape of the
  student's candidate scoring. Scoring each
  of N moves against the whole context would cost N times as much: at an
  assumed 500 moves, about a second per turn for ply one alone. The limit is
  that a move's score can use only what the summaries picked up; the recorded
  options are in the context for them to read. Exchanges and pass are
  candidates in the same list, scored by the same scorer, as the student
  scores them (the `is_play` flag); a separate exchange head was considered
  and rejected ([review record](#review-record)). The move-set runtime takes
  up to 4,096 rows per chunk
  ([model_specs.h](../../engine/include/nn/model_specs.h)), so blank-heavy
  lists run in extra chunks rather than being cut.
- **Draw queries** at a chance node. The output is a proposal distribution
  over draws. It starts as the uninformed prior and stays there until M4
  ([Build order](#build-order)). Draws are sampled from the proposal, and both
  log-probabilities go into the chance-step token.
- **Pick queries**, one per root candidate. The output is the candidate's
  value. The final pick is the argmax.

M1 needs only pick queries. Move queries are built with M3a and draw queries
with M4, where each is first used.

**Why causal attention.** This was chosen over bidirectional recomputation for
three reasons.

- **Cost.** Each decision reads a KV cache: linear in the context, not
  quadratic.
- **Training efficiency.** A recorded turn is one sequence, and one forward
  pass produces the loss at many decision points and prefix lengths, as in
  language-model training ([the training graph](#the-training-graph)).
- **Replay.** A token's representation never changes once it is computed, so
  the record, with its tick ids, defines every context the model read.

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
prior, and each draw's uninformed-prior and proposal log-probabilities. It is trained to
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
  carries that history as tokens.

  This depends on the label definition. Today's survey sims seat the
  opponent's leave only under face-up leaves; with leaves hidden they draw
  the opponent's whole rack from the unseen pool
  ([slog_position_simmer.cpp](../../engine/src/sim/slog_position_simmer.cpp)).
  A label made that way depends only on the uninformed prior, so it carries
  no signal for the history tokens to explain. **True-rack labels** need a
  sim mode that seats the opponent's full recorded rack in every rollout of
  every candidate. [sim_runner.cpp](../../engine/src/sim/sim_runner.cpp)
  already seats a known opponent leave and refills around it, so the mode is
  a small extension; it is part of M1b ([Build order](#build-order)).
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
  one sample from the posterior. It is unbiased, but its variance is the
  spread between racks, which more rollouts per candidate cannot reduce; only
  more positions can. The position count is sized before M1b, from the same
  estimate that sizes the reader's corpus ([Build order](#build-order)).
- **There is a baseline to beat.** The ported inference makes a comparison
  arm, with draws sampled from its posterior and a reader trained over them.
  Its log-probability can also go into the chance-step token as a hint the
  network is free to ignore.

## Information sets

A probe that models the opponent must model what the opponent knows, not what
is true. A fishing play shows why. Alice needs to draw for a bingo in one part
of the board, and she also needs Bob not to block it. So she plays her one or
two dump tiles elsewhere, in a way that makes Bob feel he must block there
instead. That can be a real threat or a bluff, and it works only because Bob
cannot see her leave. If Bob's modeled policy is whatever does best in this
game, it converges on exploiting Alice's actual leave, knowledge Bob does not
have. The decoy then looks worthless, because the modeled Bob always sees
through it. This is strategy fusion, the known flaw of searching over
determinized worlds: a player's policy has to be a function of their
information set, not of the hidden state.

### Where the requirement lands

SupremeBot's values do not come from backing up the moves its probes play.
They come from the reader, trained against labels, and the probes are
experiments ([Probes are experiments, not
samples](#probes-are-experiments-not-samples)). The writer that chooses Bob's
moves in a probe is rewarded for informative experiments, not for modeling
Bob, and an experiment chosen with knowledge of our leave is still a valid
experiment. So information-set correctness is required of the two things that
define value:

- **The labels.** Reply-searched labels ([The writer](#the-writer)) run nested
  sims for Bob's reply. Played from our true leave, they encode an omniscient
  Bob, and a reader trained on them values the decoy at nothing. So the nested
  sims draw our leave from Bob's belief. On the standard track this is
  required from the first reply-searched labels (M3b); under face-up leaves,
  deferred draws make it automatic.
- **The evidence the reader gets within the turn.** To value the decoy, the
  reader has to know what Bob's best response is from where he stands. That
  evidence comes from **counterfactual probes**, below.

The writer's own moves are not masked. An earlier draft masked every move
query to its mover's information set; the plan review showed that masking the
queries alone does not work, because in a causal transformer every token's
deeper layers have already read the private tokens before it, and the root
board encoding itself reads our rack. Doing it properly means computing each
token under a mask of its own viewer class at every layer, splitting action
steps into public and private parts, and giving Bob his own board encoding.
That is a separate context per viewer in all but name. It is the fallback,
built only if M3b's fishing-decoy check fails without it.

**The test for both.** Change our hidden leave while holding Bob's
observations and the sampling randomness fixed: Bob's reply in the label
generator must not change. The same paired-world test gates the fallback, if
it is ever built.

### Counterfactual probes, and traveling up the tree

A counterfactual probe replaces our leave with one drawn from **Bob's
belief**, and plays on from there. Its outcomes say what Bob's options are
worth in the worlds Bob thinks possible, which is what his response depends on.
The probes sit in the one context beside the ordinary ones, and a chance-step
flag says which leave is counterfactual.

Bob's belief comes from traveling up the tree to our root decision. After
candidate move m:

P(our leave | m) ∝ P(we play m | m's tiles + that leave) · P₀(leave)

where P₀ is the uninformed prior. The likelihood needs a model of our policy
at the root, evaluated on counterfactual racks:

- **To start:** the static-equity likelihood of the ported rack inference
  (`EquityLikelihood`, [belief/move_likelihood.h](../../engine/include/belief/move_likelihood.h)),
  which computes exactly this. Its `RackPosterior`
  ([belief/rack_inference.h](../../engine/include/belief/rack_inference.h))
  samples the posterior directly, with a uniform variate for common random
  numbers, so counterfactual leaves are drawn from it rather than from P₀. The
  chance step records both log-probabilities, as for any other proposal draw.
- **Then:** the student's policy over the root's legal moves on each
  counterfactual rack, one trunk pass per rack, batched.

The world is sampled in a fixed order, so the log-probabilities describe it
exactly: Bob's rack from our unseen pool; then our counterfactual leave from
Bob's posterior over his unseen pool; then the bag from what remains. Our
true leave's tiles return to the pool, and a tile-conservation assertion
checks every world.

This is the same computation as our own inference about Bob from his past
plays ([Rack inference is a draw decision](#rack-inference-is-a-draw-decision)),
with the seats exchanged, so one policy model serves both.

**How deep the reasoning goes.** Bob models us as the prior, not as
SupremeBot. That is level-one reasoning: a decoy has value exactly when the
prior would make that play while holding a real threat. A Bob who knew that
SupremeBot bluffs, and a SupremeBot that knew Bob knew, is equilibrium
reasoning over belief states, the territory of ReBeL and Student of Games,
and it is not planned. Generational training raises the level cheaply: as
self-labeling improves the prior (M5), Bob's model of us improves with it.

### Face-up leaves: deferred draws

Under face-up leaves the problem nearly disappears. Each leave is public, so
leave bluffs such as the fishing decoy do not exist, and the only hidden tiles
are fresh draws. A draw is chosen by the bag, not by a player, so the
opponent's belief about it is the uninformed prior: exact, and independent of
anyone's policy. No inference, no traveling up the tree, and no
counterfactual probes are needed.

What remains is omniscience about draws: a modeled Bob whose reply depends on
the tiles Alice just drew. It is removed by construction with the principle
of deferred decisions: **a player's draw is sampled when that player next
decides, not when the rules say it happens.** Bob never observes Alice's draw
before his move, so sampling it after his move changes nothing he could
condition on, and the joint distribution of the draws is unchanged. The root
already works this way: the opponent's hidden refill is the first thing a
probe samples after our move.

Two guards keep deferral exact:

- **Exchanges.** Legality uses the bag count net of pending draws, and an
  exchange resolves every pending draw before its tiles return to the bag.
  In the real game the earlier draw came first, so it can never receive the
  exchanged tiles.
- **The bag emptying.** Pending draws are resolved before the bag would empty
  and before the endgame solver takes over, when both racks become known. The
  bag count every token shows is net of pending draws.

With deferral, no decision in a face-up probe can depend on tiles its player
has not seen. That holds for the writer, for our own later moves, and for the
reply-searched labels, whose nested sims defer draws too. The paired-world
test ([Where the requirement lands](#where-the-requirement-lands)) checks it
cheaply. The rest of this section applies to the standard track only.

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

- Pick queries are applied at sampled leaf positions, so one sequence
  supervises the reader at many budgets from one probe to the full record.
  The loss is value regression plus a pairwise ranking term, and the regret
  of the argmax is the headline metric.
- **Labels, face-up leaves (M1a):** large-budget averaging simulations over
  every shortlisted candidate, the target stream of
  [sim_labeled_candidates.md](sim_labeled_candidates.md), with the opponent's
  known leave seated as the survey sims already do.
- **Labels, standard Scrabble (M1b):** the same sims with the opponent's full
  recorded rack seated ([Rack inference is a draw
  decision](#rack-inference-is-a-draw-decision)).
- Both carry their rollout policy's bias: they never learn of a YEET that
  hasty would not play
  ([simulation_information_flow.md](../simulation_information_flow.md#what-the-sideways-flows-need-from-the-model)).
  So they can teach the reader to match a large-budget averager at a fraction
  of its budget, but not to beat it.
- **Subset assembly** is valid while the writer is fixed: probes are then
  independent given the root, so any subset of a turn's probes, in any order,
  is a context the deployed agent could have produced. This multiplies rows,
  though no computation is shared between rows from one turn. Once the writer
  reads the context, probes depend on the earlier ones, and rows must be the
  recorded sequences as they were produced. This is the same constraint as
  rack_conditional_evidence.md's "targets from a context-conditioned policy".

**The label noise floor.** A label is itself an average, and its standard
error bounds what the regret curve can show. Take 16 candidates at four times
a 2,000-probe budget: 500 rollouts each, about ten times what the reader sees
per candidate, so the label's standard error is about a third of the
reader's. Past that point the curve measures label noise. Before M0, existing
survey data gives each label's standard error and the per-position labeling
cost (the survey's confirm pass is 5,000 rollouts per candidate). From those
come the largest budget at which regret stays measurable and the number of
positions M1 needs.

### The training graph

Queries are not appended to the context at deployment, so in training they are
inserted into the sequence under a mask: each query attends to the context
tokens of ticks before its own, and no token attends to a query. The mask is
block-causal by tick, which is why the record stores tick ids.

Query counts dominate the sequence. Pick queries after every one of 2,000
leaves with 16 candidates each would be 32,000 query tokens against a
context of 60,000 or more, and each M3a decision adds its summary queries and
a full legal list. So each row carries pick queries at a sample of leaf
positions, and move queries at a sample of decisions, with their lists
regenerated by the engine. Standard fused attention kernels do not take a mask of
this shape; FlexAttention's block-sparse masks do. A masked forward pass on one
synthetic 20,000-token row is prototyped before M0 fixes the record format.

### The record

A training row cannot be rebuilt by replaying moves, the invariant of
[architecture.md](../architecture.md). Its inputs include the prior's ranks
and values at every node, both draw log-probabilities and the leaf model's
readings, and recomputing them means rerunning the student and leaf model
across thousands of probes. So the record stores its inputs, and the
invariant is waived for it. The full move lists are the exception: the
engine regenerates them exactly, so they are recomputed as the invariant
intends. Regeneration depends on the lexicon and the generator, so the record
header carries the lexicon hash and the move generator's version, and the
training loader rebuilds each sampled decision's board and calls the engine.
That call's throughput is part of the training-graph prototype. Every record carries the versions of the prior
and the leaf model that produced it. A corpus is invalid for a reader
deployed with a different prior, because the prior's outputs are what the
reader learns to calibrate against. Corpus generation therefore waits until
its prior is frozen; records made earlier are for pipeline shakeout only.

### The writer

- **Start fixed.** At M1 the writer is hasty at every node, with draws from
  the uninformed prior: no trunk pass per probe, and no move queries.
- **An independent improvement signal first.** The hasty-policy labels cannot
  recognize a reply hasty misses. Trained against them, a probe that
  discovers YEET and correctly overturns the root ranking is scored as a loss,
  because the label disagrees with it. So before the writer trains, the
  labels must be able to see what the writer is meant to find. **Reply-searched
  labels** do that: at each labeling rollout's ply one, the opponent's reply
  is the best of a shortlist by nested sims, not hasty's argmax. The nested
  sims draw our leave from the opponent's belief, not the true one, and pass
  the paired-world test ([Information sets](#information-sets)). That is
  expensive, but it is paid for labels only. Self-labeling by a larger-budget
  SupremeBot is a second such source, and it waits until the writer has shown
  it finds replies the labels missed.
- **Then reinforcement learning.** The writer's purpose is to make the pick
  better. Define the potential of a prefix as the label value of the reader's
  current argmax, measured by a pick query placed immediately after each leaf
  token, in recorded order. A leaf's reward is the change in potential it
  causes. The rewards telescope over leaf tokens: summed over a turn, they
  equal the final pick's label value minus the prior's pick's. The reward is
  credited to the decisions of the probe that produced the leaf. A decision
  whose value lies in steering later probes, not in its own probe's leaf, gets
  no direct credit this way; that is a known limit ([Open
  questions](#open-questions)). The definition is checked on a toy bandit
  before M3a. This needs a label for every candidate the reader may pick, so
  during training the pick is restricted to labeled candidates.
- **Alternate the two roles.** A new writer changes the reader's input
  distribution, so the reader retrains on the new writer's turns before the
  writer steps again, in the manner of AlphaZero's generations
  ([generational_training.md](../generational_training.md)).

### Budget generalization

A learned search is only as good as the budgets it was trained at. Rows span
budgets from zero to beyond the deployment budget, up to the label noise
floor, and the headline curve is regret against budget. It must keep falling
past the largest training budget. A curve that flattens there means
SupremeBot has learned a budget-specific routine rather than how to search.

## Cost

Reading the context is cheap. Here is an order-of-magnitude estimate under
stated assumptions, to be replaced by the throughput microbenchmark
([Build order](#build-order)):

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

Recorded options ([Move lists](#move-lists-local-and-global)) grow the context
to 60,000–85,000 tokens. Reading stays cheap, about 100 ms per turn at peak at
85,000, because attention per decision is linear in the context. **Memory is
what binds:** the KV cache grows to about 260 MB per turn at 85,000 tokens,
and self-play holds many turns at once, about 8 GB for 32 concurrent turns.

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
fall back to hasty with its cheap generation. Start with ply one and the
probe's first own move, and widen the scope when a measurement says it pays.
At ply one, the scope also costs one trunk pass per probe
([What an action step costs](#what-an-action-step-costs)).

**The serving runtime.** Nothing in the engine serves this model today.
`NeuralNet` is a synchronous TensorRT wrapper with one execution context
([neural_net.h](../../engine/include/nn/neural_net.h)). It has no KV cache, no
incremental append, and no way to query without appending, and concurrent
self-play games would each need their own cache. M1 does not need it: its
writer is fixed, and its reader can run one full forward pass at pick time.
M3a does, so the runtime is a work item of its own, with the choice between
TensorRT with dynamic KV bindings and in-process PyTorch serving made there.
Whether M3a is affordable at all is settled earlier, by a throughput
microbenchmark alongside M0: a random-weight model at the planned width and
depth, real tick batching, full generation at the queried nodes and the
ply-one trunk pass, reported as probes per second against hasty.

## Risks

**No free convergence.** MCTS improves with more search even with a bad
network, because its backup rules are sound. SupremeBot has no such rules. If
the reader has learned nothing useful about a region of positions, more probes
do not help there. The floor ([The network](#the-network)) guarantees the
prior's quality, not improvement over it. The budget curve is the guard.

**Transfer can go wrong in both directions.** The reader may fail to transfer
what should transfer (ZIT to QUIZATH), or transfer what should not (ZIT to a
rack without a T). Both are measured directly at M1 with synthetic contexts
built to contain exactly one such fact.

**A reader can beat averaging without transferring anything.** Shrinking each
candidate's probe mean toward its prior already beats plain averaging at small
budgets. So M1's kill criterion is measured against a shrinkage estimator, not
against averaging.

**Throughput.** If full move generation, the round trips and the deep-node
boards hold probes per second far below hasty rollouts even at ply-one scope,
the learned writer is not affordable. SupremeBot then reduces to M1's learned
reader over cheap probes. That is still a scheme with every sideways valuation
flow, and it reaches rack_conditional_evidence.md's destination by a different
route.

**Opacity.** When a known case fails, there is no node table to inspect. A
bug looks exactly like "the model did not learn". Each milestone therefore
comes with its own diagnostic: identical-record comparisons and synthetic
single-fact contexts at M1, and attention attribution from the pick query back
to the probes that moved it.

**The labels' bias.** Until self-labeling starts, labels inherit hasty's
blind spots, and the reader learns to reproduce them faster, not to remove
them.

## Build order

Each milestone produces a working agent, measured before the next begins. The
work runs in two tracks. The **face-up track** tests the core bet, transfer
and learned steering, where deferred draws make information sets free. The
**standard track** then adds hidden leaves and the machinery they need. The
destination is standard Scrabble; the order defers its complexity until the
core has been shown to work.

**The face-up track**

- **Before M0: three estimates.** The label noise floor and labeling cost
  from existing survey data, which sets the training budgets and M1's corpus
  size ([The reader](#the-reader)). The masked training graph prototyped on
  one synthetic row ([The training graph](#the-training-graph)). The
  throughput microbenchmark ([Cost](#cost)), which decides whether M3a is
  affordable. Both run at the grown context, 85,000 tokens, with concurrent
  turns and the TensorRT models resident, and report peak GPU memory as a
  gate beside probes per second. With them come two move-list counts, the
  typical legal-list length per node and how fast each shared board's
  recorded set stops growing as racks accumulate
  ([Move lists](#move-lists-local-and-global)), and the cost of
  reply-searched labels: outer rollouts × reply shortlist × inner rollouts,
  times the position count the noise-floor estimate asks for. If that is
  unaffordable, M3a cannot be funded, and the plan must know before M0.
- **M0: the record.** Per-step probe logging, with the fields in
  [Tokens](#tokens), tick ids, the prior and leaf-model versions, and the
  lexicon and move-generator versions. Draws are deferred
  ([Face-up leaves: deferred draws](#face-up-leaves-deferred-draws)). This
  builds rack_conditional_evidence.md's layer 1, which was never built,
  generalized from per-rollout to per-step. Plus the token encoder.
- **M1a: learned reader, fixed writer, face-up leaves. The kill gate.** It
  uses the existing face-up prior, so it waits on no retraining. The writer is
  hasty at every node, with draws from the uninformed prior, which is exact
  under face-up leaves. The reader is measured on **identical records**:
  every arm values the same probes, so the comparison isolates the valuation.
  The report is regret against budget, in three arms:

  | arm | valuation |
  |---|---|
  | reader | learned |
  | shrinkage | per candidate: the prior and the probe mean, combined with fitted variances |
  | averaging | mean per candidate |

  Ply-one options are recorded by the static-equity rule
  ([Move lists](#move-lists-local-and-global)). Beside the arms run the
  synthetic single-fact transfer tests, built as production records with
  option tokens: QUIZETH to QUIZATH, the no-T control, a near miss (one tile
  short) and a blank-bearing case.
  *Kill criterion:* if the reader does not beat shrinkage on identical records
  at matched budgets, or fails the transfer tests, transfer is not being
  learned: stop. If it passes, match play against BestBot under face-up
  leaves.
- **M2: the known positions** that exist under face-up leaves, among them the
  ACETA family in `positions/NWL23/interesting-positions/`.
- **M3a: learned move choices.** The serving runtime ([Cost](#cost)), then
  reply-searched labels with deferred draws ([The writer](#the-writer)), then
  the writer, trained with the telescoping reward and alternating with the
  reader. Measured in match play against M1a at equal wall-clock time, not
  equal probes, because steering costs time.
- **M5: self-labeling.** SupremeBot at many times the budget labels
  SupremeBot's training positions, once M3a has shown it finds replies the
  labels missed. It applies again on the standard track.

**The standard track**

- **From now, in parallel: the standard-Scrabble prior.** The teacher,
  student and move proposal model retrained with `face_up_leaves` off. This
  is new tags, not new code, and it is ready long before the track needs it.
- **M1b: the reader in standard Scrabble.** It needs the standard-Scrabble
  prior, true-rack labels and the opponent-history tokens. It adds the
  inference arms:

  | arm | valuation | draws |
  |---|---|---|
  | reader | learned, with opponent history | uninformed prior |
  | reader, history ablated | learned, without opponent history | uninformed prior |
  | shrinkage | as in M1a | uninformed prior |
  | shrinkage with inference | as in M1a | the ported posterior ([belief/rack_inference.h](../../engine/include/belief/rack_inference.h)) |

  The full reader against the ablated one measures the implicit inference,
  and the last arm is what that inference has to match. If the reader passes,
  match play against BestBot.
- **M3b: information sets.** Reply-searched labels with leaves drawn from the
  opponent's belief, and the paired-world test; then counterfactual probes
  ([Information sets](#information-sets)), measured against the version
  without them on a known fishing-decoy position. The Richards–Johnson
  position joins the known set here, since its read depends on hidden leaves.
- **M4: learned draws.** Proposal distributions at chance nodes: the rack
  inference ([Rack inference is a draw decision](#rack-inference-is-a-draw-decision)),
  measured against M1b's shrinkage-with-inference arm. It comes after learned
  moves because it distorts the reader's input distribution the most.

## Open questions

- **Deep-node boards:** move deltas over the root board, or a trunk encode per
  node. M1 uses deltas; the encode variant is built only if M1 looks
  representation-limited.
- **The recorded subset:** k, the region-diversity slots, and whether the
  writer should learn what to record.
- **Masked opponent contexts:** whether M3b's fishing-decoy check needs them
  (the fallback in [Information sets](#information-sets)).
- **Counterfactual probe share:** how much of the budget models the
  opponent's view, and at which plies.
- **How deep the opponent reasoning goes:** level one (the opponent models us
  as the prior) is planned; equilibrium over belief states is not.
- **Opponent history:** every past opponent turn, or only the last few.
  RackInferrer conditions on the last move only.
- **Credit for steering:** how a decision whose value is in changing later
  probes, not its own leaf, gets credit.
- **The serving runtime:** TensorRT with dynamic KV bindings, or in-process
  PyTorch.
- **Summary tokens:** whether causal revision needs them, and if so, their
  schedule and what trains them.
- **Stopping:** a fixed budget first. A learned stop head fits the same
  query mechanism and the telescoping reward, less a cost per probe.
- **Tick size:** the batching staleness against throughput.
- **Context across turns:** discarded at the end of each turn, as in every
  scheme so far. Carrying over the probes that remain legal is deferred until a
  case needs it.
- **The reader's implicit posterior before M4:** how close it comes to the
  ported posterior, M1b's fourth arm, and so how much M4 has to add.

## Review record

Plan review, 2026-09-29: four independent panelists (hidden complexity, rival
design on a different vendor's model, scope, integration). Every blocking and
serious critique, and each minor one, with its resolution:

| Critique | Resolution |
|---|---|
| **Blocking.** The first labels cannot teach implicit inference: with leaves hidden, the survey sims draw the opponent's whole rack uniformly, so M1's history arms would converge by construction. | Revised. Verified in `slog_position_simmer.cpp`. True-rack labels are defined (a small sim-mode extension), the history arms move to M1b where those labels exist, and the label noise is sized first. |
| **Blocking.** Every action-step token asks for the prior's rank at its node, which is the per-node trunk pass the design claims to avoid, at ply one included. | Revised. The fields depend on the node (the prior inside the query scope, static equity outside it); the ply-one trunk pass is costed; M1's writer is hasty everywhere. |
| No stronger rival than averaging: a hybrid that keeps explicit, probability-weighted valuation over a shared, periodically rebuilt latent would give global transfer without asking one reader to learn probability correction and value together. | Partly revised: the cheap core of that rival, per-candidate estimation that uses the prior, is M1's shrinkage arm and the kill criterion's bar. The full hybrid is close to rack_conditional_evidence.md, which the direction has put on hiatus, so it is not built. **Human call, decided 2026-09-29:** no hybrid gate. |
| The writer would train against hasty-biased labels, which score a correct YEET discovery as a loss; self-labeling cannot correct that later. | Revised. Reply-searched labels come before the writer trains, and self-labeling waits until the writer has shown it finds replies the labels missed. |
| The standard-Scrabble retrain, the longest step, sits in front of the milestone that can kill the project; the kill test does not need hidden racks. Raised by two panelists. | Revised: M1a runs on face-up leaves with the existing prior; the retrain runs in parallel and M1b follows. **Human call, decided 2026-09-29:** confirmed, and extended: the whole core (M0 to M3a) runs on face-up leaves, where deferred draws make information sets free, before the standard track. |
| The history ablation does not isolate transfer: a history-free reader can beat averaging by calibration alone. | Revised, together with the next row. |
| The kill criterion passes through shrinkage toward the prior, with no transfer. | Revised. The criterion is "beats shrinkage", and the synthetic transfer tests move into M1a. |
| roadmap.md still describes the evidence-loop agent as active, contradicting the plan's status. | Revised. roadmap.md is rewritten in the same PR. |
| No decision on reusing the item-3 runtime for the root prefix. | Revised. The root prefix reuses the cache graph through `MoveProposalService`; the fusion step graph is not used. |
| Storing the prior's outputs breaks the replay-reconstruction invariant and ties each corpus to one prior version. | Revised. [The record](#the-record) waives the invariant with its reason, versions every record, and orders corpus generation after the prior is frozen. |
| "One forward pass trains every decision" hides a block-causal mask by tick and query counts larger than the context. | Revised. [The training graph](#the-training-graph): masked query layout, sampled queries, FlexAttention, and a prototype before the record format is fixed. |
| No milestone builds the KV-cached serving runtime, and the throughput fact that decides M3 arrives late. | Revised. The runtime is part of M3; a throughput microbenchmark runs before M0. |
| Deterministic ticks need a barrier that the endgame solver stalls; batched BF16 inference does not reproduce decisions. | Revised. Slow probes leave the tick loop; replay replays recorded choices and tick ids. |
| The telescoping reward is undefined when probes interleave across ticks. | Revised. The reward is per leaf token, measured in recorded order; credit for steering is an open question; the definition is checked on a toy bandit before M3. |
| The label noise floor bounds the budget curve, and the labeling cost was never estimated. | Revised. Both are estimated from existing survey data before M0 and set the budgets and corpus size. |
| Minor: the history tokens are underspecified. | Revised: a token spec; how many past turns is an open question. |
| Minor: history tokens and the inference arm serve a variant M1 does not need. | Revised in effect: they are used from M1b. The history tokens are still built in M0, so the record format is fixed once. |
| Minor: move and draw queries are specified before any milestone uses them. | Revised: built with M3 and M4. |
| Minor: M1 builds two deep-node representations. | Revised: move deltas only, unless M1 looks representation-limited. |
| Minor: the sampled-distribution log-probability is constant until M4. | Rejected: it costs one field, and keeping it avoids a record format change at M4. |
| Minor: "extends layer 1" implies layer 1 exists. | Revised: M0 builds it. |

Plan review, round 2, 2026-09-29, of the move-list, exchange and
information-set additions: the same four seats (hidden complexity, rival
design on codex, scope, integration).

| Critique | Resolution |
|---|---|
| **Blocking.** Masking the move queries does not enforce information sets: every token's deeper layers have already read the private tokens before it, and the root board encoding reads our rack. Raised by two panelists; the rival proposed separate per-viewer contexts that share weights. | Revised, by relocating the requirement. Values come from labels and the reader, not from the writer's moves, so information-set correctness is required of the labels (belief-drawn leaves, a paired-world test) and supplied to the reader as counterfactual-probe evidence. Writer masking is dropped. The properly built version (viewer classes at every layer, public/private token split, a Bob-view board) is the documented fallback if the fishing-decoy check fails. **Human call to confirm:** this changes the mechanism from the one agreed in conversation, though it keeps traveling up the tree. |
| No objective makes a masked opponent policy a best response over his belief; the writer is rewarded for informative experiments, which benefit from reading our leave. | Revised: this critique is the reason for the relocation above. |
| **Blocking.** Options deduplicated per board carry leave and static equity, which depend on the rack that first selected them. Raised by three panelists. | Revised. Options are rack-free (placement, tiles, tile count, score); the reader derives leave from the querying rack, as `MoveFeatureArrays` does. |
| Selection by the prior does not match the static-equity field stored. | Revised: selection is by static equity plus region slots, and the token stores neither value. |
| **Blocking.** M1a's transfer test needs options, but M1's hasty writer has no prior to select them by. | Revised: the static-equity rule works at M1 without a trunk pass; the synthetic tests are built as production records with options, plus near-miss and blank cases. |
| Record options only from M3, and hand-inject them into M1a's synthetic tests. | Rejected: M1a's reader would then be tested on a mechanism it never trained on. Options are recorded from M0, ply-one boards only at M1. |
| Keep lexical facts in an engine-managed, board-keyed index with retrieval, instead of the causal context. | Rejected in favor of rack-free options plus containment, which is exact linear algebra given the tile count; retrieval would add misses and a second store. The index's separation of rack-free and rack-dependent facts is adopted. |
| The separate exchange head reverses the closed A4 verdict, breaks the empty-context floor, and diverges from the single-scorer `is_play` convention. Raised by three panelists. | Revised: exchanges and pass are candidates in the one scored list; they are still never recorded as options. |
| The two-level readout's summaries need the node's trunk tokens, and list lengths exceed the runtime's 512-row cap. | Revised in part: trunk tokens are a side input at in-scope nodes. The cap does not exist: 512 is the optimization profile, the maximum is 4,096 rows per chunk, and longer lists take extra chunks. |
| The prototypes and microbenchmark are sized to the old 20,000-token context. | Revised: 85,000 tokens, concurrent turns, models resident, peak memory as a gate. |
| The reply-searched label cost is never estimated. | Revised: added to the pre-M0 estimates, with its consequence for M3 stated. |
| Split counterfactual probes out of M3's first slice. | Revised: M3 has two slices; the belief-drawn labels stay in the first, since without them the labels encode an omniscient opponent. |
| Minor: counterfactual leaves drawn from P₀ waste probes. | Revised: drawn from `RackPosterior`. |
| Minor: the joint sampling order of a counterfactual world is unspecified. | Revised: order fixed, tile conservation asserted. |
| Minor: containment is a dot product and a comparison, and blanks relax it. | Revised: tile count in the token; blank rule stated; tests added. |
| Minor: regeneration needs lexicon and generator versions, and a loader-side engine call. | Revised. |
| Minor: cite the existing full-list precedent. | Revised: UltimateBot's root. |
| Minor: split the exchange head into its own step. | Moot: no exchange head. |
