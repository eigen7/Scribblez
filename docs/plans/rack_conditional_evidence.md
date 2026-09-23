# Rack-conditional evidence: transferring sim knowledge across the tree

## Purpose

Simming a candidate move `M` samples opponent racks and plays them out. Now
and then a sampled rack `R` reveals that the opponent has a reply far better
than the one the hasty rollout policy plays. That matters for `M`'s valuation,
but the important question is the next one: was that specific to `R`, or is
there a fact here that holds across the opponent's whole rack distribution?

A human answers the second question routinely: "as long as he has an F or an
M, he has that lane", "a Q without a U is a liability here", "six of the seven
letters for the B-lane bingo and he fishes". Having found the fact once, the
human applies it everywhere it holds: to the other candidates, to the racks
not yet sampled, and to what the opponent would do in every rollout from now
on. Discovering it independently in each rollout is what the sims do today,
and it is prohibitively expensive.

This document extends the sim-residual loop
([sim_residual_feedback.md](sim_residual_feedback.md)) so that knowledge found
in one part of the turn's search transfers to the rest of it:

- **Forward**: rollouts after candidate `DOG` play replies informed by what
  the sims of `CAT` found.
- **Backward**: rollouts already run for `CAT` are re-priced, or re-run where
  it matters, once the sims of `DOG` show they used an inferior reply.
- **Sideways**: candidates never simmed are valued per rack from what the
  simmed ones revealed, which is what promotes the modest move that fixes a
  losing rack region.

The value of this in Scrabble is high for a structural reason: the random bag
gives the tree an enormous branching factor, and large swaths of those
branches are strategically alike because what makes them alike is the board,
which every branch shares.

## Where this sits

- **[sim_residual_feedback.md](sim_residual_feedback.md)** is the base. Its
  evidence token is an *aggregate* over a candidate's rollouts, and the loop
  it drives closes a candidate once simmed. This plan keeps its principles
  (prediction-beside-observation, late fusion, permutation-invariant evidence
  sets, subset-assembled training rows, the CRN pairing) and changes the
  granularity: the evidence keeps its racks. It does not keep the
  implementation of that plan; [what the existing stack
  loses](#what-the-existing-stack-loses) is the inventory.
- **[design.md §8.1](../design.md)** (search-derived knowledge buffers) is
  the idea in the abstract. Here the buffer is the evidence set itself; there
  is no separate table of discovered facts.
- **[roadmap.md item 6](../roadmap.md)** built UltimateBot over the base loop.
  The agent described below replaces its decision procedure.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)** and
  [blind_spots.md](../blind_spots.md) collect the positions where hasty-policy
  sims fail. Those positions are this plan's evaluation set.

## What is exact and what is learned

In Scrabble the thing a human generalizes over is unusually concrete. The
opponent's rack is a multiset of at most seven tiles over 27 types. Whether a
sampled rack holds a given multiset is set inclusion. Whether a given reply is
still legal after some other candidate is a single-move legality check, a few
lookups per placed tile. The probability that the opponent holds given tiles
is an exact hypergeometric over the unseen pool, sharper under face-up leaves
where the retained tiles are known. Under common random numbers (CRN,
[sim_runner.h](../../engine/include/sim/sim_runner.h)) rollout index `i` has
the same opponent rack for every candidate, and that rack is reproducible from
the seed. None of this needs a model.

What is learned is the **outcome as a function of the rack**, at this board,
after this candidate. Every human predicate above is a region of that
function's domain: a union ("F or M"), a conjunction with an absence ("Q
without U"), a near-miss ("six of seven"). The predicates are never
enumerated. A rack encoder trained across many positions learns whatever
partition of rack space explains the outcomes, and the within-turn evidence
updates that prior in the regions the rollouts actually cover.

That division has a coverage limit that must be stated plainly. Rollouts
sample racks from the natural distribution, so a region one rack in five
hundred falls into contributes two rollouts to a thousand. The within-turn
evidence cannot resolve it; the amortized prior has to carry it, from the
many positions where the same kind of structure recurred. Where the prior is
weak the engine can supply cheap **lexical primitives** without naming
predicates: for the few lanes the sims flag as hot, enumerate the words that
fit once, and give each rack its maximum letter overlap with any of them.
That makes "six of seven" a one-dimensional feature the encoder composes
freely.

## The evidence set at rollout granularity

The unit of evidence is one token per **rack index**, not per rollout or per
candidate. Index `i` carries the sampled rack, and for every candidate rolled
on that index:

- the reply the rollout played and our follow-up;
- the horizon outcome (WLD and the score-difference moments, root-mover POV,
  from the leaf model or the terminal state as today);
- the model's **prior prediction** for that candidate on that rack, so the
  residual is formed per rack inside the model, for the reason the base plan
  gives (a posterior that scales the prior needs the prior as an input);
- the **gap**: the current conditioned score of the reply that was played
  minus the score of the reply now believed best (the section on outdated
  rollouts below), recomputed every block;
- a provenance flag: whether this outcome has been superseded by a re-run on
  the same index.

A token per index keeps the set at the rollout count, not rollouts times
candidates, and it places the cross-candidate contrast for one rack inside
one token, which is what the attention needs to read.

A second token kind records **nested sims**: at a spot-checked reply node, the
outcomes of several replies on one rack after one candidate. These are the
"discovered replies" of the human's reasoning. They go into the same set, so
a verified reply is evidence, not an entry in a side table, and the reply
policy applies it in-context to candidates it was never verified against.

The three-way fact retained from the base plan: evidence stays paired with
its move, the set is order-free, and an empty set must reduce the model to the
plain one-pass student.

## The model: three readers of one context

One evidence-conditioned move set evaluation model, the base plan's student
plus its fusion stage, is read at three places. Late fusion remains the
load-bearing constraint, and the cost structure below depends on it.

1. **Root valuation and acquisition.** For any candidate, simmed or not, the
   model predicts the outcome on each context rack. Its paired mean against
   the incumbent over the same racks, with the model's predictive spread, is
   the expected gain of the base plan's proves-best head with a rack axis.
   A candidate no round proposed rises when the context shows it fixes the
   rack region where the incumbent bleeds. This needs a head no model has
   today: the student sees the unseen pool, never the opponent's rack, and
   returns one outcome per candidate. The **rack-query head** takes a
   candidate's fused latent and one opponent rack and returns that
   candidate's outcome on that rack; it reads the latent, never the raw
   context, so a query costs a small MLP or a short attention over the
   latent, not a context read. Queries are still numerous (legal moves times
   active racks per block), so per-rack scoring runs only on a shortlist
   taken from the per-candidate prediction.
2. **The reply policy at ply one of every rollout.** The board after `DOG` is
   the same for every rollout of `DOG`; only the replier's rack differs. The
   design wants the trunk encode and the fusion of the context into it to run
   **once per candidate per block**, with per-rollout work reduced to the
   reply move list and a cheap scoring pass of that list against the cached
   conditioned encoding. **Today's student cannot do this.** Its trunk reads
   the mover's rack three ways: the rack counts, the unseen-pool thermometer
   (the pool minus the mover's rack), and the opponent-reach plane gated on
   that pool ([game_state_encoder.cpp](../../engine/src/encoding/game_state_encoder.cpp),
   `encode_board` in [model.py](../../py/scribblez/move_set_eval/model.py)).
   At ply one the mover holds a different rack in every rollout, so the
   trunk input differs per rollout. Two ways through:

   - **Per-rollout encodes, batched.** Correct with today's student, at
     rollouts times contenders trunk passes per block. Acceptable for
     measuring the ply-one policy's strength; too slow as the design center.
   - **A rack-late student.** The trunk reads the board and the
     rack-independent scalars; the rack, the pool it implies and the reach
     plane enter at scoring. This needs a re-distilled student and a
     quality gate against the current one, and readers 1 to 3 all assume it.

   Either way the ply-one policy needs the full reply list, which greedy
   hasty does not produce: its WordMap search stops without enumerating the
   legal plays ([macondo_bot.h](../../engine/include/agent/macondo_bot.h)),
   and full generation costs about twice as much. Scoring also needs a GPU
   round trip in the middle of a rollout, which nothing downstream can
   defer the way the horizon readout is deferred. So ply one is
   **block-staged**: generate every rollout's reply list for the block,
   score them in one batch, then resume each rollout with its first move
   forced.

   This is where `CAT`'s
   discoveries reach `DOG`'s rollouts: on a rack with an M, the conditioned
   student scores the lane play above the hasty move although nobody rolled
   that reply after `DOG`. With an empty context the conditioned student is
   the plain student, which already outranks equity ranking, so hasty was
   never the ceiling for ply one. Deeper plies stay hasty for now: after the
   reply the board differs per rollout, and a learned policy there needs a
   trunk encode per rollout.
3. **The correction for outdated rollouts.** The value gap between the
   post-reply states of the played reply and the preferred one, read from the
   student's per-move value heads.

## What the existing stack loses

The base plan's per-candidate aggregate token is load-bearing across a
shipped surface, and this plan replaces most of it rather than extending it.

| Piece | Today | Under this plan |
|---|---|---|
| Evidence unit | `EvidenceSet`: one `SimObservation` per simmed candidate ([move_proposal_service.h](../../engine/include/agent/move_proposal_service.h)) | Rack-index tokens, each a variable-size set of per-candidate sub-records, plus nested-sim tokens. The sub-record encoder is new. |
| Fusion | Self-attention over at most 64 padded evidence tokens ([evidence_fusion.py](../../py/scribblez/evidence_fusion.py)) | Cross-attention into board tokens or inducing points, over a context of thousands. The exported graph needs a dynamic evidence axis. |
| Evidence encoding | Board tokens gathered from the root board's map, cached once | Encoded against each candidate's post-move board. |
| Serving | `MoveProposalService` holds one encoded position | A session holding each contender's fused encoding at once, for reader 2. |
| Agent loop | `evidence_loop.h`: one sim call per candidate, one observation back | Blocks over (candidate, rack indices), block-staged ply one. |
| Training | `py/scribblez/evidence/` and `py/scribblez/sim_evidence/`, keyed to per-candidate observations | Per-rollout targets, rack-query rows, reply-node rows. |

The aggregate path stays in service until layer 3 beats it on the known
cases, then retires; the two do not interoperate. Each new piece lands in the
layer that first needs it (build order, below).

## The turn

**State.** The legal move list, with the trunk output and move encodings
cached once. The context set. The active rack indices. For every rollout ever
run: its candidate, rack index, cached reply list, the reply played, the
outcome.

**One seed stream, shared by every candidate.** The seed stream defines rack
`i` for all candidates. Every simmed candidate is rolled on every active
index; a newly proposed candidate rolls on the whole active set; a top-up
activates new indices for every contender still in play. Pairing is never
broken among moves being compared. Contenders that are clearly out stop
receiving top-ups, which breaks pairing only for moves no longer compared.

**The first sim** is the mechanical anchor, the highest-raw-score candidate,
as in the base plan.

**Each block:**

1. **Re-fuse.** Fuse the current context into each contender's cached trunk
   output.
2. **Re-price.** Rescore every existing rollout's cached reply shortlist
   against its candidate's new encoding and update its gap. The shortlist is
   the top few replies by the plain score, kept at generation time; caching
   every rollout's full list is hundreds of megabytes per turn, and the gap
   is computed over the shortlist. A slight change of mind
   that flips the argmax between near-equal replies produces a gap near zero
   and the rollout stays nearly as good as fresh; a large gap says the
   rollout understates the opponent by about that much. This pass is linear
   in rollouts, touches no trunk, and reads no context.
3. **Choose the next block of work.** Every option has one shape: a set of
   rack indices crossed with a candidate. Re-running `CAT`'s high-gap
   rollouts, adding fresh racks to `DOG`, and giving `EMU` its first
   rollouts are the same kind of option, ranked by one acquisition rule, the
   expected improvement of the final decision. A candidate with no rollouts
   scores through its model prediction under the current context with
   maximal uncertainty, which is what puts it in the race. There is no
   "simmed" state: a candidate has some rollouts, some gap on each, and some
   model uncertainty on the racks that decide the pick.

   The unified rule needs an epistemic uncertainty the model does not
   produce: its heads give outcome distributions, not how sure the model is
   of them. Until a source is chosen and validated (an ensemble, or a
   variance head trained against held-out rollouts), the loop runs a fixed
   heuristic schedule: re-run the highest-gap rollouts on the floor indices,
   top up the contenders the root ranking separates least, and admit the
   next candidate by the proves-best gain as today.

   Which new indices to activate is adaptive: look ahead at the next few
   dozen racks in the stream, score each by contender disagreement times
   model uncertainty, and activate the best ones plus a **floor** of
   unscored natural draws, so no rack region can go dark. Adaptive
   activation biases the raw paired mean over the active set toward the
   regions the agent chose to look at; the estimate that absorbs this is the
   model's mean over a fresh natural rack sample (below), and the raw
   fallback mean uses the floor indices only.
4. **Run the block; add its tokens.** Nested spot checks inside the block add
   tokens of the second kind. A re-run replaces the outcome on that index
   and flags the superseded one rather than deleting it.

**Inside one rollout.** The opponent's rack for index `i` comes from the
seed; the reply is the conditioned student's choice over the generated reply
list (reader 2). On a capped number of reply nodes per turn, placed where the
context shows a high-residual rack region and with an off-policy floor, a
**spot check** runs a small nested sim over the top few replies at a shallow
horizon and records the result as a nested-sim token. Our rack is our leave
plus a draw from the shared shuffled bag order; our move and the remaining
plies to the horizon use the hasty policy. The bag emptying hands the endgame
to the solver as today.

**Scoring.** At the horizon the position evaluation model reads the mover's
post-move pre-draw state and returns WLD and the score-difference Gaussian,
flipped to the root mover's POV; a game that ends earlier contributes its
exact result. The outcome does not go into a running mean. It becomes a
token, beside the prior prediction for this candidate on this rack.

**Stopping.** The loop is at a fixed point when no block's expected
improvement clears the threshold, with the budget as the outer bound. The
reply policy can flip back and forth on some rack as the context grows; an
oscillating reply is a diagnostic to log, not something to suppress.

**The final pick.** Each contender's value is the conditioned model's mean
over a fresh natural rack sample of several thousand racks, conditioned on
the full context: free, since it needs no rollouts, and paired across
contenders on the same sample so rack luck cancels there as it did under
CRN. Pick the highest on the winrate objective. Two guardrails. A move must
have rollouts on the floor indices before it can be chosen, so a raw paired
mean exists as the fallback and no move is played on a model's promise. That
mean is paired only if every contender's floor rollouts used the same reply
policy. Under the loop they do not: a candidate simmed late faces a
better-informed opponent than one simmed early, and the difference is
systematic. So before the pick, the floor indices of every eligible
contender are re-run under one policy snapshot, and those re-runs are
charged to the budget. And
until the adjusted estimate has proven calibrated in match play, a
disagreement between adjusted and raw rankings beyond the raw standard error
is logged as a case to study.

**Across turns.** The context does not persist: its racks were drawn from a
pool that no longer exists once the opponent moves. Nothing else needs to.

## Outdated rollouts: correct or re-run, never discard

A rollout whose reply the policy would now choose differently is not
worthless, and treating it as such is importance weighting under a
deterministic policy, whose weights are all zero or one. The estimator should
be model-based and continuous.

- **Correct it.** Observed outcome plus the value gap between the two
  post-reply states (reader 3). The model half of a doubly-robust estimator.
  Right when the better reply's advantage is realized immediately: a higher
  score, a better leave.
- **Re-run it.** Play the preferred reply and the plies after it. Right when
  the advantage is realized later, a setup or a block whose payoff the
  one-step value estimate carries only approximately, or when the gap
  estimate is itself uncertain, which it is whenever the preferred reply is
  a fresh discovery with few nested-sim tokens behind it.

The acquisition rule prices both: a re-run's value is the expected reduction
in decision uncertainty from replacing a corrected outcome with an observed
one, high when the gap is large and the correction uncertain and the rack
lies where the contenders' rankings turn, near zero for a small gap.

**The correction trains itself.** Every re-run yields a pair, same rack, same
candidate, one outcome under each reply. That pair is the supervised target
for the correction, so the agent's own re-runs calibrate the model that lets
it skip re-runs elsewhere. Keeping the superseded token is what makes the
pair available.

Outdated rollouts also give two things for free: the played reply is one the
opponent could make, so its outcome bounds their options from below; and
pairing still holds across contenders whose rollouts on that rack used replies
of similar quality, which is most of them, since a policy shift usually
affects a rack region rather than one candidate.

## Training

The row structure of the base plan survives. A position's pool gives context
subsets over its simmed candidates; the held-out candidate's rollouts give
per-rack targets on the same rack indices; the loss is per rollout rather
than per aggregate. Rows are assembled at every context size from empty up to
the deployment budget, with sub-sampled indices, so the model learns to read
the partial contexts it sees at every block before the last.

**Reply-node rows.** A spot-checked reply node is a position with a context
attached. Its nested sim gives outcomes for several replies on one rack after
one candidate; the root's context set is attached, and the target is the
usual ranking by sim outcome. These rows teach the transfer directly, because
the context holds rollouts from sibling candidates on overlapping rack
regions. Whether the model generalizes the idea across candidates rather than
memorizing move identities is settled the same way the base plan settles its
placement-plane bet: train with and without the sibling candidates' tokens in
the context, measure at the reply node.

**Correction rows** are the re-run pairs above.

**Where context length bites.** A row with a ten-thousand-token context is
expensive to backpropagate through, and pools produce combinatorially many
such rows. Subset assembly at many sizes is the answer, and rack-index tokens
keep the count at rollouts rather than rollouts times candidates.

**Off-policy drift.** The deployed context distribution is produced by the
agent's own loop, and adaptive activation, re-runs and corrections push it
away from anything assembled from static pools. This is handled
generationally, as the base plan handles the proposer, and it is a reason to
add those loop features one at a time (build order, below).

**Targets from a context-conditioned policy.** Subset assembly is sound in
the base plan because a rollout's outcome does not depend on the context: the
reply policy is hasty. Once replies come from the conditioned student, a
held-out rollout's outcome depends on the context that was live when it ran,
and a row that pairs it with some other subset asks the model to predict a
policy whose input it cannot see. So every rollout records the reply-policy
model and the context version that chose its reply, and the first corpora
use only context-free reply policies, hasty or the plain student. Rows whose
targets came from a conditioned policy carry that policy's context, not an
arbitrary subset, and they wait until layer 4 needs them.

## Cost accounting

Two different quadratics have to be kept apart.

**Per-rollout re-reading, the LLM pattern.** A language model reads its
whole context once per generated token, so generating `N` tokens costs
`O(N²)` even with a KV cache. The analogue here would be a rollout that reads
the context, and none does: a rollout reads a cached, fixed-size fused
encoding of its candidate's board. The context is read only when that
encoding is rebuilt, once per block per contender, so the number of context
reads is the number of blocks `B`, tens, not the number of rollouts `N`.
This is why late fusion is load-bearing: evidence that modulated the trunk or
the per-rollout scoring pass would have to be read per rollout.

**Inside one read.** Whether a read is linear or quadratic in the context
size depends on the fusion architecture. Self-attention among the evidence
tokens, as the base plan specifies, makes one read `O(n²)`, and summed over
blocks the total grows as `N²·B/3`. Cross-attention only, the context
attended into the board tokens or into a small fixed set of inducing points
(a set transformer), makes one read `O(n·L)` and the total `O(N·B)`. This
plan takes the linear form, for a reason beyond cost: the rack-conditional
readout is kernel regression, a query rack weighting context racks by
similarity, which is cross-attention. The base plan's argument for pairwise
self-attention, contrasts between candidate pairs, is already served by
placing every candidate's outcome for one rack inside one token. The raw set
is kept and the latent readout is recomputed from all of it each block, so
this is not the lossy recurrent memory the base plan rejects.

For scale: even a full self-attention read over ten thousand tokens is on the
order of a hundred million pairs per layer, tens of milliseconds for a few
bf16 layers on a 4090, against seconds for the ten thousand rollouts. The
quadratic read would not bind at today's budgets; the linear form is chosen
so that it never does.

Per rollout, the added work over today is at ply one: full reply
generation, move-feature encoding of that list, and its share of the
block-staged scoring batch, plus a trunk encode if the student is not
rack-late (reader 2). This term, not the context read, is what bounds
rollouts per second, and it is measured before the design commits to it.
Per block, the re-pricing pass scores each rollout's reply shortlist and
touches no trunk. The rack-query head adds shortlisted candidates times
active racks small queries per block, and several thousand racks times the
contenders at the final pick. The root proposal over all legal moves is one
scoring pass per block, the same shape as today's proposer.

## Risks

**The knowledge representation is more constrained than it looks.** The
learned object is a residual over a 27-dimensional count space on top of the
plain student, read from about a thousand paired samples; that is in-context
regression on a small input, and with empty-context rows in training the
floor is the current system. Every token field is an exact quantity.

**The loop dynamics are not constrained.** The conditioned policy changes the
rollouts, which change the context, which changes the policy. The corrections
use the same student that chose the replies, so a wrong preference looks
self-consistent until a re-run contradicts it; re-runs are the external check
and the leaf model is fixed, which is what keeps that check honest. And there
are enough moving parts that a bug looks exactly like "the model did not
learn".

**The deficiency cannot be measured under the policy that hides it.** An
earlier draft of this plan proposed a first gate: stratify existing rollouts
by letter presence and see whether rack composition explains outcome variance
beyond noise. That gate is withdrawn. Its outcomes come from hasty rollouts,
and if the deficiency is that hasty opponents do not exploit their racks, the
outcomes cannot show the structure; a null there would measure the instrument,
not the problem. The existence of the deficiency is settled by the expert
cases and by hasty's construction. What is open is which fix removes it.

## Evaluation and build order

**Three fixes, and the classification that chooses between them.** The
unrealistic hasty policy inside rollouts is a known deficiency, but in-context
transfer is one of three fixes for it, and the other two are cheaper:

- a **stronger static rollout policy**, the plain student at ply one, with no
  in-turn machinery, for failures of the kind "the simmed opponent never
  fishes, never sets up, never defends";
- a **deeper reply search** at a sampled fraction of reply nodes, averaged
  in: position-specific, no learning, expensive but simple;
- **in-context transfer**, this plan, for failures that are position-specific
  in a way a static policy cannot learn and a reply search cannot afford to
  rediscover per rollout.

Before building, take the known failure positions (the blind-spot
collections and `positions/NWL23/interesting-positions/`) and for each ask
what the simmed opponent did wrong and what the minimal machinery is that
would make it do right. If most fall to the first bucket, a better rollout
policy is the whole first step and the rest of this plan waits. If a
meaningful share need the third, those positions are the mandate.

**What is measured.** The known cases, directly: does the agent find the
reply the experts say it should, and choose the root move they say is right.
And match play against the previous version, because a fix that gets the
examples right can still lose on the distribution. Not statistical proxies
that can miss the effect.

**Layers, one at a time.** The sequencing is a debugging discipline, not a
hedge on the idea: built at once, a known case that still fails points at
nothing. Each layer is checked against the same cases before the next.

1. **Per-rollout logging.** The `.sobs` record gains, per rollout, the rack
   index, the reply played, our follow-up, and the outcome. The runner has
   all of it transiently today. Small next to the count planes.
2. **The plain student as the ply-one reply policy.** No context. This is
   the first bucket's fix and the floor for everything after; it alone may
   move the known cases.
3. **The rack-conditioned evidence model**, static contexts, natural racks,
   no loop: trained on subset-assembled rows with per-rollout targets,
   compared with the aggregate-token model on the held-out candidate's
   per-rollout outcomes, and at the root on the known cases.
4. **The conditioned reply policy** at spot-checked nodes, static contexts
   still, nested sims as truth; the sibling-token ablation above.
5. **The loop**, one feature per step, each measured in match play against
   the version without it: plain re-runs first (exact and cheap under
   truncation), then model-based corrections, then adaptive rack
   activation. Adaptive activation buys variance reduction, not knowledge,
   and is the feature that most distorts the training distribution, so it
   goes last.

## Open questions

- **Budget split** between candidate proposals, rack top-ups and re-runs;
  whether the spot-check cap is a fixed count or a fraction of rollouts.
- **How far the learned policy extends into the rollout.** Ply one is the
  cut where the trunk encode is per candidate; extending to our own follow-up
  costs an encode per rollout and is measurable.
- **Nested-sim shape**: how many replies, how many rollouts, how shallow a
  horizon, before a spot check is worth its cost.
- **Whether the conditioned student generalizes ideas across candidates**
  rather than memorizing move identities (the sibling-token ablation).
- **Context persistence across turns** for the nested-sim tokens only: their
  replies are board-conditioned and a legality check would make stale ones
  harmless. Deferred until there is a case that needs it.
