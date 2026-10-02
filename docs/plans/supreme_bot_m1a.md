# SupremeBot M1a: implementation plan

**Status: proposed; nothing built.** This is the build plan for M1a, the held-out
transfer test that is SupremeBot's kill gate. The test itself (what is
measured, the arms, the metrics, the controls and the kill criterion) is
specified in [supreme_bot.md](supreme_bot.md#the-transfer-test-m1a); this
document says how to build it.

**Goal.** A corpus of face-up positions, each with 16 stratified and coupled
candidates, hasty probes recorded step by step, and large-budget labels from
the same estimator; a reader trained on it; and an evaluation harness that
produces the kill-criterion table.

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
| Horizon | 3 plies (our move, the reply, our next move, then the leaf). | Deeper horizons, which cost more per probe without changing what M1a tests. |
| Couplings | Play vs exchange of the same tiles; the same tiles at two footprints; the same lane with one tile different. All are exact matches in the legal list. | The hot-lane coupling, deferred until a hot lane has a definition. |
| Running it | A dashboard workload from the start, on the `move_set_eval` pattern: a generate role running the C++ tool, a train role later. | A standalone script. |

## Step 0: the noise and saturation estimate

This step sets the corpus size, so it runs first. Using PR 1 and a measurement
mode of PR 2's tool, on about 300 face-up positions, run the labeling
estimator (3-ply truncated hasty, the frozen teacher as leaf) on all 16
candidates at a large rollout count, recording **paired** per-rollout
differences under common random numbers. It reports:

- per-candidate and paired standard errors against rollout count, and from
  them the label rollout count L at which within-row differences of the
  expected score are resolvable;
- rollouts per second, and so the cost per position and the corpus size a
  given compute window affords;
- how quickly the ply-one options saturate: the union of each probe's
  static-equity top k on a shared board, against probe count;
- the count of each coupling kind per position.

## PR 1: probe traces in the engine

In [sim_runner](../../engine/include/sim/sim_runner.h):

- A rollout variant returning a `RolloutTrace` beside the `Rollout`: the
  opponent's sampled rack; per turn, the mover, move, rack before, bag count
  before, score change and tiles drawn; the leaf's win/draw/loss and score
  readings before reduction; and whether the rollout ended at the horizon or
  at the game's end.
- Ply-one options: on each probe's opponent rack, `equity_top_k` at ply one,
  keeping the top k plus the region slots. This is the only full move
  generation; every later ply stays greedy hasty.
- No per-step static-equity rank. Hasty's move is always its own rank 1, so
  the field carries nothing until the writer is learned.
- Tests: a trace reproduces its `Rollout` exactly, and traces are
  identical across thread counts.

## PR 2: the generator and the record format

A new tool, `transfer_test_generator`, and its workload:

- **Positions.** Sampled from the existing face-up `.slog` corpora with
  `sample_eligible_turns` and `position_seed`, stratified by game phase, split
  into train and test by game.
- **Candidates.** The full legal list with equities
  (`generate_legal_plays` and `generate_legal_exchanges`
  ([agent.h](../../engine/include/agent/agent.h)), scored by
  `HastyEquity::equities`) feeds a selector for the strata (6 top, 5 middle, 3
  exchanges, 2 low, by hasty equity) and the coupled pairs. Each candidate is
  tagged with its stratum and coupling id.
- **Labels.** L rollouts per candidate through `SimRunner`, reduced to
  `RolloutStats` and written as a `.sobs` v5 file with a new flag marking it
  as labels.
- **Probes.** P probes per candidate, written to a new **`.sprobe`** sidecar:
  - file header: magic, version, the face-up flag, the teacher and leaf
    hashes, the horizon, the lexicon hash and the generator version;
  - per position: the candidates with their stratum and coupling tags;
  - per probe: the candidate index, the rollout index, the opponent's rack
    and the turn records from the trace;
  - per shared board: the recorded options.

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
  - leaf: the win/draw/loss and score readings;
  - option: move features, tile count, and its board.

  Every token also carries its candidate slot (shared with the candidate
  token), its depth and its type.
- **Row assembly.** Hold out a random subset H of one to four candidates
  across strata, drop H's probes, subset-assemble the rest (random subsets and
  orders), and place pick queries at sampled prefix lengths. The graded
  variant keeps one to five of each held-out move's probes.

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
PR 1 ── PR 2 ──┬── step 0 ── corpus run (labels and probes)
               ├── PR 3 ── PR 4 ── PR 5 ── verdict
               └── PR 6
```

PRs 3 to 5 develop against a small shakeout corpus while the real one
generates.
