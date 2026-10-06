# SupremeBot M1a: implementation plan

**Status: step 0 and PRs 1 to 4 built, and the corpus generated (2026-10-05);
PRs 5 and 6 proposed, not built.**
This is the build plan for M1a, the held-out
transfer test that is SupremeBot's kill gate. The test itself (what is
measured, the arms, the metrics, the controls and the kill criterion) is
specified in [supreme_bot.md](supreme_bot.md#the-transfer-test-m1a); this
document says how to build it.

**Goal.** A corpus of face-up positions, each with 16 stratified and coupled
candidates, hasty probes recorded step by step, and labels from the same
estimator at a larger budget; a reader trained on it; and an evaluation
harness that produces the kill-criterion table.

## Starting point

What exists, and what that changes:

- **Per-rollout traces exist but are discarded.** `run_rollout`
  ([sim_runner.cpp](../../engine/src/sim/sim_runner.cpp)) builds a full
  `GameLog`: the opponent's sampled rack, and per turn the mover, the move,
  the rack before it, the bag count and the tiles drawn. It returns only
  `Rollout` (the reply, our next move, the leaf reading). Recording
  probes is a change to what that function returns, not a new simulator.
- **Labels already have a record.** `RolloutStats`
  ([sim_runner.h](../../engine/include/sim/sim_runner.h)) holds win, draw and
  loss counts, the score-difference sum and sum of squares (so its mean and
  standard deviation), and 2927-class footprint histograms for all four
  placement heads. Labels are written as `.sobs` records; no new label format
  is needed.
- **Value truncation exists** through `SimRunner::Params` (a horizon of at
  least `kMinHorizonPlies = 3` and a leaf service), used today only by
  `sim_obs_tool`.
- **Common random numbers cover the opponent's first rack.** Rollout i of
  every candidate draws the same opponent rack; later draws diverge, because
  each candidate refills a different number of tiles.
- **No face-up student or move-proposal export exists on disk.** The face-up
  teachers do, with checkpoints and ONNX exports, for example
  `position_eval/transformer-clipped` (epoch 2543).
- **No trained summary-token model exists.** The evidence-loop model was
  never trained to a usable checkpoint, its footprint heads were never
  trained, and the only `.sobs` files on disk are version 1, which the
  current reader rejects.
- **The face-up corpus is large.** The `position_eval` tags under
  `/workspace/mount/tags` hold about 3.9M face-up games, about 81M eligible
  positions, readable with
  [slog_sampling.h](../../engine/include/data/slog_sampling.h).

## Decisions

| Decision | Choice | Instead of |
|---|---|---|
| The prior | The face-up **teacher**, evaluated on each candidate's post-move position: exact per-candidate win/draw/loss, score and footprint predictions. Its trunk on the root position supplies the root board tokens. | The student, which does not exist on disk and only approximates the teacher. The student returns when the prior must score every legal move. |
| One frozen teacher | The same checkpoint is the prior, the leaf model and the root encoder, and its hash is stamped in every record. | Separate models, which would add version skew for no benefit at M1a. |
| The summary-token arm | The reader's own architecture fed one aggregate token per probed candidate (win/draw/loss frequencies, score moments, footprint histograms) instead of per-probe tokens. | The evidence-loop model, which would have to be resurrected along with a student. The ablation answers the same question, whether probe content matters beyond outcomes, with identical capacity and training. |
| Candidate strata | Ranked by **hasty equity**: CPU-only selection. | Ranking by the teacher. |
| Horizon | 3 plies after the candidate (the opponent's reply, our next move, the opponent's next move), then the leaf. | Deeper horizons, which cost more per probe without changing what M1a tests. |
| Couplings | Play vs exchange of the same tiles; the same tiles at two footprints; the same lane with one tile different. All are exact matches in the legal list. | The hot-lane coupling, deferred until a hot lane has a definition. |
| Running it | A dashboard workload from the start, on the `move_set_eval` pattern: a generate role running the C++ tool, a train role later. | A standalone script. |
| Labels | Training positions at **L = 100–200** rollouts per candidate; a **test set of 1,000 positions at L = 1,000**; label rollouts on seeds independent of the probes' ([step 0 results](#step-0-results)). | 1,000 rollouts everywhere, which costs ten times as much per training position for noise the headline can absorb. |
| Options | **None recorded in M1a.** | Ply-one options by the static-equity rule, which step 0 found do not saturate and the tested transfer does not need ([step 0 results](#step-0-results)). |
| Corpus | 10,000 training positions and the 1,000-position test set to start, grown if M1a's comparisons lack power. | A size fixed in advance, before the reader's advantage is known. |

## Step 0: the noise and saturation estimate

This step sets the corpus size, so it runs first. On about 300 face-up
positions, it runs the labeling estimator (3-ply truncated hasty, the frozen
teacher as leaf) on all 16 candidates at 10,000 rollouts each, and keeps every
rollout. It reports:

- the label rollout count L at which within-position differences of the
  expected score are resolvable: per position, the noise variance of each
  candidate's label, centered per rollout on the position's candidates, against
  the spread of the labels. Every smaller rollout count is read from the same
  rollouts. It is reported over every candidate and over the top and middle
  strata alone: exchanges and low-ranked plays sit far below the best plays and
  dominate a spread over every candidate, and the comparisons that are hard to
  resolve are among the plausible moves;
- rollouts per second, and so the cost per position and the corpus size a
  given compute window affords;
- how quickly the ply-one options saturate: on each candidate's board, the
  union of the opponent's static-equity top k over the racks the rollouts dealt
  them, against probe count;
- how many coupled pairs of each kind a position offers, and how many the
  selection takes.

**As built.** The `transfer_test` workload runs `transfer_test_generator
--mode=measure`, which the corpus modes of PR 2 extend. Each cycle self-plays a
face-up hasty batch and measures one position per game, rather than sampling
another tag's corpus: the self-play is negligible next to the sims, and the
games come from the same policy. Candidates come from
[transfer_candidates.h](../../engine/include/sim/transfer_candidates.h). Each
`.slog` gets two sidecars: `.tmeasure` (JSON: parameters, timings, and per
position its candidates, couplings and saturation curves) and `.trollouts`
(every rollout's expected score and score difference, as float32).
`py/scripts/transfer_test_measure_report.py --tag TAG` prints the report.
`Rollout` now records the opponent's sampled rack, which PR 1's traces need
too.

### Step 0 results

Tag `transfer_test/m1a-step0`, 2026-10-03: 300 positions, 16 candidates each,
10,000 truncated rollouts per candidate with the `transformer-clipped` epoch
2543 leaf. Seven positions are decided endgames, every rollout the same
outcome, and are left out of the expected-score figures.

**Label noise.** Among the plausible moves (the top and middle strata), the
expected score spreads with a median standard deviation of 0.037 within a
position (interquartile 0.024 to 0.055). A label's standard error at L
rollouts, and its noise variance against that spread:

| L | standard error | noise / signal variance, median | p75 |
|---|---|---|---|
| 100 | about 0.010 | 0.087 | 0.167 |
| 200 | about 0.007 | 0.043 | 0.084 |
| 1,000 | 0.0033 | 0.009 | 0.017 |
| 2,000 | 0.0023 | 0.004 | 0.008 |

Label noise is unbiased: it adds the same floor to every arm's error, costing
statistical power, not correctness. At L = 100 a training position costs a
tenth of one at 1,000, and ten times the positions buys more power than the
noise takes away, which is also what training the reader wants. The test set
is small, so it can afford L = 1,000 and a headline nearly free of label noise.
Because L = 100 is no larger than the probe budget, label rollouts must use
seeds independent of the probes', or a reader could match the labels' noise.

The best and second-best plausible moves differ by a median 0.0137, and a
quarter of positions by under 0.004: near-ties, where choosing either costs
almost nothing. At L = 1,000 the top two are not separated at two paired
standard errors in 37% of positions, at 2,000 in 32%. That limits the secondary
decision readouts, not the headline, and the gap-weighted ranking term already
allows for it.

**Common random numbers** shrink the per-rollout variance of the centered
expected score by a median 1.8x: modest, because only the opponent's first
rack is shared.

**Cost.** 3,203 rollouts per second on 28 threads. With about 125 probes per
candidate (2,000 per position), a training position at L = 100 is 3,600
rollouts, about 1.1 s, and 10,000 of them about 3.1 hours (4.5 hours at
L = 200). A test position at L = 1,000 is 18,000 rollouts, about 5.6 s, and
the 1,000-position test set about 1.6 hours.

**Ply-one options do not saturate.** On a plausible move's board, the union
of the opponent's static-equity top 16 over the racks the probes dealt grows
by about 1.5x per doubling of the probes: about 950 distinct options at 128
probes, 1,460 at 256. At a realistic probe count that is about 15,000 option
tokens per position, and the per-board deduplication saves only about 3x.
M1a records none, because the transfer it tests does not need them:

- the headline concerns held-out moves, whose boards are never probed and so
  would have no options anyway;
- containment transfer such as QUIZETH to QUIZATH needs no options either: a
  probe's action steps carry each played move's tiles and its chance steps
  carry each rack, so whether a played move fits another rack is a check
  between tokens the context already holds. Options add only moves no probe
  played.

An options ablation can follow if transfer comes out weak.

**Couplings are plentiful.** A play-exchange pair is offered in 92% of
positions (the rest have fewer than seven tiles in the bag), with a median 97
play-exchange, 258 same-lane and 463 same-tiles pairs per position. The
selected sixteen hold on average 1.75 play-exchange, 2.79 same-lane and 4.87
same-tiles pairs, counting the ones that arise among the stratified picks.

**A first look at transfer.** With no learned model: after removing what
static equity predicts (65% of the within-position spread of the full-budget
sim values), the leftover sim values of two moves correlate as follows:

| pair | correlation | pairs |
|---|---|---|
| random siblings | -0.12 | 1,336 |
| same tiles, two placements | +0.21 | 1,418 |
| same lane, one tile different | +0.17 | 825 |
| play vs exchange of those tiles | +0.04 | 524 |

The baseline is negative because centering sixteen values on their mean
anti-correlates unrelated ones. Moves sharing tiles or a lane sit about 0.3
above it, well beyond the 0.03 to 0.04 standard error: simming one tells you
something about the other, though at a correlation near 0.2 one partner
explains only about 4% of the other's leftover variance. Play against exchange
transfers least, plausibly because static equity already prices the leave
they share. This is a lower bound on what the reader can find: a weaker prior
than the teacher, one partner, linear, and outcomes only.

## PR 1: probe traces in the engine

**As built.** `SimRunner::run_rollouts` takes an optional vector of
`RolloutTrace`, filled at each rollout's index: the rollout's turns as the game
log records them, and whether it stopped at the horizon. The leaf readings are
the `Rollout` beside it.

In [sim_runner](../../engine/include/sim/sim_runner.h):

- A rollout variant returning a `RolloutTrace` beside the `Rollout`: the
  opponent's sampled rack; per turn, the mover, move, rack before, bag count
  before, score change and tiles drawn; the leaf's win/draw/loss and score
  readings before reduction; and whether the rollout ended at the horizon or
  at the game's end.
- No options and no full move generation: every ply stays greedy hasty
  ([step 0 results](#step-0-results)).
- No per-step static-equity rank. Hasty's move is always its own rank 1, so
  the field carries nothing until the writer is learned.
- Tests: a trace reproduces its `Rollout` exactly, and traces are
  identical across thread counts.

## PR 2: the generator and the record format

A new tool, `transfer_test_generator`, and its workload:

- **Positions.** As in step 0: each cycle self-plays a face-up hasty batch and
  takes one position per game. The test set is its own tag, so train and test
  never share a game.
- **Candidates.** The full legal list with equities
  (`generate_legal_plays` and `generate_legal_exchanges`
  ([agent.h](../../engine/include/agent/agent.h)), scored by
  `HastyEquity::equities`) feeds a selector for the strata (6 top, 5 middle, 3
  exchanges, 2 low, by hasty equity) and the coupled pairs. Each candidate is
  tagged with its stratum and coupling id.
- **Labels.** L rollouts per candidate through `SimRunner` (100 to 200 for
  training, 1,000 for the test set), on seeds offset past the probes' so the
  two never share rollouts, reduced to `RolloutStats` and written as a `.sobs`
  v5 file with a new flag marking it as labels.
- **Probes.** P probes per candidate, written to a new **`.sprobe`** sidecar:
  - file header: magic, version, the face-up flag, the teacher and leaf
    hashes, the horizon, the lexicon hash and the generator version;
  - per position: the candidates with their stratum and coupling tags;
  - per probe: the candidate index, the rollout index, the opponent's rack
    and the turn records from the trace.

  Packed structs, published through
  [format_layout.cpp](../../engine/src/data/format_layout.cpp), with a
  golden-pin test.
- **All candidates are probed.** Holding moves out happens at row assembly,
  so one corpus serves every held-out split, the graded variant and the
  partner ablation.
- **The workload** follows
  [move_set_eval.py](../../py/scribblez/workloads/move_set_eval.py): the tool
  joins the worker bundle, and the sidecars are delivered through the pair
  store. The step-0 measurement mode is a flag on the same tool.

**As built.** `transfer_test_generator --mode=corpus`, and the
`transfer_test` workload's `train-corpus` (L = 100, 10,000 positions, the
default) and `test-corpus` (L = 1,000, 1,000 positions) profiles; `measure` is
step 0. Where it differs from the list above:

- **The `.sprobe` keeps to the replay rule.** M1a stores no prior outputs, so
  a record holds what a `.slog` would: the racks each side began the rollout
  with and each turn's move and draw (`TurnBlob`), plus the outcome. A turn's
  mover, racks, bag count and score come back by replay from the position
  after the candidate; PR 3 reads them through the FFI, as `.slog` rows are.
  The header carries the leaf model's hash, the lexicon's name, the horizon
  and the probe count. Format: [probe_log.h](../../engine/include/data/probe_log.h).
- **Coupling tags are not stored.** They are a pure function of the
  candidates (`find_couplings`), so readers recompute them; each candidate
  stores its stratum.
- **Labels reuse the probe pass's position and candidates.** The labels run a
  second `SimRunner` on the replayed position and the candidates the probe
  pass selected, seeded at `base_seed + probes`, so label rollout i is rollout
  probes + i and never one of the probes. The `.sobs` carries
  `kSimObsFlagLabels` and is written before the `.sprobe`, whose presence marks
  a file done.

## PR 3: the data side of the reader

- A numpy reader for `.sprobe`, on the pattern of
  [targets.py](../../py/scribblez/move_set_eval/targets.py); labels through
  [sobs.py](../../py/scribblez/sim_evidence/sobs.py).
- **The prior cache.** Once per position: the frozen teacher's root trunk
  tokens (board rows from `ffi.decode_rows`) and each candidate's post-move
  predictions, meaning win/draw/loss, the score mean and standard deviation,
  and the four footprint heads. Cached beside the records.
- **The token encoder.** One input projection per token type into width C:
  - candidate: `ffi.encode_moves` features, the prior's predictions, a
    stratum embedding;
  - chance step: the drawn tiles and the resulting rack, as cumulative counts;
  - action step: move features against the root board, leave counts, bag
    count, mover and depth;
  - leaf: the win/draw/loss and score readings.

  Every token also carries its candidate slot (shared with the candidate
  token), its depth and its type.
- **Row assembly.** Hold out a random subset H of one to four candidates
  across strata, drop H's probes, subset-assemble the rest (random subsets and
  orders), and place pick queries at sampled prefix lengths. The graded
  variant keeps one to five of each held-out move's probes.

**As built.** The package `scribblez.transfer_test` (probes, prior, corpus,
rows, tokens) and `py/scripts/transfer_test_prior.py`. Where it differs from
the list above, or settles what the list left open:

- **The replay is the engine's.** `data/probe_replay` replays each probe from
  the position after its candidate with the `.slog` turn-replay rules
  (`binlog::replay_turn_records`, now startable from any state), and the FFI
  serves the result per file: the root, each candidate's leave, score and
  bag, each probe's two opening deals (the root mover's refill, the
  opponent's rack beyond their known leave), and per turn the mover, ply,
  bag, score difference, leave, draw and rack. Racks cross as tile codes. A
  record whose racks do not follow from its position fails the replay. A
  100-position file replays in about 0.1 s.
- **The prior comes from the teacher's torch checkpoint.** The root tokens are
  the trunk's hidden state, which the ONNX export does not expose. The tag's
  rolling checkpoint must hold the pinned generation, and before any cache
  is written its value heads are checked against that generation's ONNX,
  the leaf model the probes ran. The check rejects a neighbouring export
  (generation 2400 is off by 1.1 in win/draw/loss logits). The cache stores
  the trunk's 225 cell tokens and its scalar projection for the root, and per
  candidate the win/draw/loss probabilities, the score mean and standard
  deviation, and the four footprint heads as raw logits. Footprint masking
  is left to PR 5, which compares footprint distributions; the reader takes
  their unmasked log-softmax. On the test corpus the prior's expected score
  correlates 0.995 with the labels.
- **The root summary is a token.** The context opens with 226 root tokens: the
  225 cells, then the scalar projection.
- **Queries are listed apart from the context.** Each names its candidate and
  the number of context tokens it may see, always a probe boundary; the
  full context is always one of the prefixes, and every candidate is asked at
  each. PR 4 appends them after the context under that mask.
- **A token budget bounds a row.** Each kept candidate keeps a uniform 0 to 32
  of its probes, the kept probes are interleaved at random, and probes are
  taken in that order while they fit 2,048 context tokens. A chance token
  follows only a turn that drew tiles.
- **Labels are slimmed** to the outcome counts, the score moments and the two
  next-move footprint histograms, about a third of a `.sobs` record.
- **Both corpora have their caches** (4.4 GB for the training corpus, under 4 minutes
  on the local GPU). The training corpus loads in 27 s at 11.7 GB resident,
  and one core assembles about 400 rows a second.

## PR 4: the reader and its trainer

- A small causal transformer with FlexAttention: pick queries attend to their
  prefix only, and nothing attends to a query. Grouped-query attention.
- **Heads per candidate:** win/draw/loss; the score mean and log standard
  deviation; a mean and spread on the expected score; the opp-next and
  self-next footprint distributions.
- **Loss:** cross-entropy on win/draw/loss and footprints against the label's
  empirical distributions; Gaussian negative log-likelihood on the moments,
  with the label variance added; the gap-weighted ranking term. The
  game-result anchor belongs to the label loop and is left out here.
- **Trainer:** the position-eval recipe (bf16 autocast, `torch.compile`,
  activation checkpointing as needed), AdamW, the rolling checkpoint helpers,
  as the workload's train role.
- **The size sweep:** width and depth flags for readers of about 1M, 5M, 25M
  and 100M parameters, on two corpus sizes.

**As built.** `scribblez.transfer_test` gains reader, loss and trainer, and
a new workload, `transfer_reader`, runs one reader per tag on a corpus tag.
Its profiles are the sweep's shapes: `reader-1m`, `reader-5m` (the
default), `reader-25m` and `reader-100m`, at 1.3M, 5.0M, 23.6M and 90.9M
parameters; `train_positions` takes a fixed subset for the second corpus
size. Where it differs from the list above, or settles what the list left
open:

- **Its own workload.** A reader tag names its corpus tag rather than sharing
  `transfer_test`'s tags, since each sweep run is a tag and generator and
  trainer parameters do not mix. Creating a tag refuses a corpus without
  prior caches.
- **Every head is a correction to the prior.** The heads answer the teacher's
  prior for the query's candidate plus a learned correction that starts at
  zero, so an untrained reader is exactly the prior and training learns what
  the evidence adds.
- **The likelihoods are beta-NLL.** Plain Gaussian NLL let the expected-score
  head lower its loss by widening its spread instead of moving its mean: the
  reader could not overfit 32 rows with it, and did with squared error.
  Weighting each term by its own detached variance (beta-NLL, beta = 1) gives
  the mean a squared-error gradient and keeps the spread learning the misfit.
- **A fixed same-candidate attention bias.** Each head adds a fixed bias, 0 to
  4 across the heads, between tokens of the same candidate. Without it the
  reader learned nothing in 3,000 steps even on a synthetic target that was
  exactly the mean of each candidate's probe outcomes in the row (error 0.064
  to 0.063); with it, the same target fell to 0.019 in 2,000 steps. A
  learned bias trained at least twice as slowly, its gradient a reduction over
  every score.
- **Token content is normalized per kind.** A candidate's thousands of prior
  placements had swamped a leaf's few outcome features; each kind's content
  is now RMS-normalized on its own.
- **Validation.** 5% of the corpus's positions, chosen by a hash of file and
  position, so every run on a corpus validates on the same ones. Each pass
  records the within-row expected-score error at the full context, on
  held-out and probed candidates, for the reader and the prior.
- **The first read**, `reader-5m` for 3,000 steps on the training corpus:
  on probed candidates the error fell from the prior's 0.0284 to 0.0233,
  past the best fixed shrinkage of probe means toward the prior (0.0256, at a
  weight of 32 probes); on held-out candidates it stayed near the prior's
  0.0259. Probe means alone do worse than the prior (0.057). All against L =
  100 labels, whose own noise is about 0.015.
- **Cost.** About 0.19 s a step for `reader-5m` with six row workers, so 20,000
  steps take about an hour; `reader-100m` needs activation checkpointing to fit
  16 GiB and takes about 1.5 s a step. Batches leave the row workers as numpy
  arrays: as tensors they would pass through `/dev/shm`, which a container
  holds to 64 MB.

## The near-endgame corpus

**Why.** The first sweep runs (`reader-1m`, `reader-5m`, 20,000 steps on the
training corpus) showed no transfer: held-out error never beat the prior
beyond noise and grew worse once the reader began memorizing the 9,479
positions' labels (training ranking loss falling to 0.053 while validation
rose to 0.139). Two findings moved the corpus to the near endgame.

- **Where the teacher is wrong.** The teacher's within-row error against the
  labels, by tiles in the bag before the move. On the training corpus (L =
  100, truncated rollouts, greedy endgames):

  | Bag | Positions | Teacher error | Label spread |
  |---|---|---|---|
  | 0-6 | 556 | 0.093 | 0.099 |
  | 7-14 | 982 | 0.040 | 0.050 |
  | 15-24 | 1,234 | 0.023 | 0.042 |
  | 25-59 | 3,949 | 0.016-0.018 | 0.040 |
  | 60-89 | 3,060 | 0.014 | 0.039 |

  Past 15 tiles the teacher's error is near the labels' own noise (about
  0.010-0.015 at L = 100), so there is little for a reader to learn there.
  On 300 near-endgame positions relabelled with rollouts played to the end
  (no leaf model) and endgames solved, the error stands: 0.078 at bags 1-3,
  0.054 at 4-7, 0.039 at 8-11, 0.033 at 12-15, against a label-noise bound of
  0.020. Greedy endgames move the labels by up to 0.05 but leave the
  teacher's error within 0.003, so it is the teacher's, not the labels'.
- **The exhibits.** Two hand-built positions where the right move blocks a
  threat that only other moves' rollouts reveal; candidates simmed with
  solved endgames and face-up leaves:
  - **pos-09** (position-eval test set): the opponent's G hook at M7 forming
    GNU, with -ING words down column M. The teacher puts the opponent on M7
    at 0.27 against the Monte-Carlo 0.67. The six column-N plays that kill
    the lane sim at 0.953 against the teacher's 0.829; the 22 moves that
    leave it open sim at 0.814 against 0.858: the teacher under-prices
    blocking by 0.17.
  - **egotize-lane** (face-up trajectory set): GAVE opens row 13 to EGOTIZE.
    The four clean blockers sim at 1.000 and the open moves at 0.909, while
    the teacher rates both groups about 0.97-0.99: an under-pricing of about
    0.09. The teacher places the opponent's reply there correctly (0.36 vs
    0.34); it misses that the reply wins the game.

**Decisions.** The `endgame-train-corpus` and `endgame-test-corpus` profiles
of the `transfer_test` workload:

- positions with 1 to 15 tiles in the bag (`min_bag`, `max_bag`), sampled
  from the same per-game order as before, filtered;
- rollouts played to the end (`horizon` 0), so probes show real endings and
  labels owe nothing to the teacher, and no leaf model, so no GPU;
- endgames solved in every rollout (`solve_max_unseen` 100): the gate is the
  root's unseen count, and the old threshold of 14 left bags 8-15 greedy;
- L = 300 for training and 1,000 for the test set, since a finished game's
  0/1 outcome is noisier than a leaf reading.

Measured cost on 28 threads at 4 probes and 200 labels per candidate: 3.2
positions a second with greedy endgames, 0.48 with every endgame solved. At
125 probes and 300 labels the 10,000-position training corpus is about 12
hours on one such machine.

## PR 5: the evaluation harness

- **Arms**, all on identical records: the teacher prior; the common shift; the
  similarity-weighted shift (kernel regression on residuals over same-leave,
  footprint-overlap, same-lane and score features, weights fitted on training
  positions); the summary-token ablation; shrinkage (graded variant only); the
  reader.
- **Metrics:** per-head plain error on held-out moves; centered within each
  position and split into row and within-row parts, with the within-row
  expected-score error as the headline; per stratum, against probe count,
  with paired bootstrap intervals over positions.
- **Controls:** the partner ablation, shuffled evidence, and similar against
  dissimilar held-out moves.
- **Output:** one report with the kill-criterion table and the secondary
  readouts.

## PR 6: the synthetic single-fact tests

Constructed `.gcg` positions (the QUIZETH/QUIZATH family, the no-T control, a
near miss, a blank-bearing case), and a generator mode that fixes the
opponent's draw per probe, so each record contains exactly the intended
fact. Evaluated by the same reader and harness. It depends only on PR 2 and
can be built at any point after it.

## Order

```
step 0 (done) ── PR 1 ── PR 2 ──┬── corpus run (labels and probes)
                                ├── PR 3 ── PR 4 ── PR 5 ── verdict
                                └── PR 6
```

PRs 3 to 5 develop against a small shakeout corpus while the real one
generates.
