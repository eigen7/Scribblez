# Sim-labeled candidate rows for the position evaluation model

A plan for closing a measured blind spot in the position evaluation model:
setup plays. It adds a second target stream to the teacher -- Monte-Carlo sim
outcomes for every simmed candidate at a sampled position, each candidate's
post-move state becoming a training row -- and, as a later phase, lets the
sim's pick steer the self-play game past the sampled turn. The roadmap has
carried the first half as a "planned second target stream" for the teacher
([roadmap.md](../roadmap.md), *The position evaluation model*); this is the
concrete route to it, motivated by one position.

## The finding

`positions/NWL23/interesting-positions/ACETA.gcg`, turn 2 (Sokol to move,
AACITTZ, after Anderson's GREEK holding EF). Sokol plays K6 AC.TA for 7: the
A's land right of the TL squares J6 and J10, so ZA is 31 at either, and the C
between them means no two-letter word ending in C can block both. The
`transformer-clipped` teacher (epoch 2500) ranks it 101st of 137 legal
placements.

Twelve candidates -- the model's top ten plus both ACETA variants -- were
simmed to a natural end, 2000 HastyBot rollouts each, under both information
conditions (`monte_carlo_sim_tool`, standard error about 1.1 points):

| play | model win% | sim face-up | sim hidden |
|---|---|---|---|
| 9K TIZ | 44.0 | 44.1 | 43.3 |
| K6 AC.TA | 33.9 | 44.1 | 43.0 |
| L3 ATTAC. | 40.8 | 41.1 | 40.8 |
| J6 AC.TA (the A's *on* the TLs) | 36.8 | 36.4 | 36.2 |

Every on-distribution candidate is priced within a point of the sim; the setup
misses by ten. It is a specific blind spot, not a calibration offset. The
placement heads show where: the model's opponent-next plane after K6 AC.TA is
right (J10-J12 hot, the FA hook, matching the sim's 41-55% coverage), but its
self-next plane puts 0.34 on the mover's Z hook where the sim's rollouts put
0.57. It sees the threat and undercounts its own payoff. The F in the
opponent's leave moves the sim not at all (44.1 -> 44.2 with the F replaced
by an O) and moves the model by 4.6 points -- the model treats the two hooks
as independent threats where the sim, like the human, sees that only one can
be taken. Reproduce the readouts with

```
neural_rank_tool --gcg positions/NWL23/interesting-positions/ACETA.gcg --turn 2 -k 0 --model <onnx>
```

(the committed file records the whole game, so `--turn 2` selects the
position; the K6 line is the played move). Add `--sim` for the sim win rate
and rank of every scored move beside the model's.

## Diagnosis: coverage, not capacity

The trained tag generated its games HastyBot-vs-HastyBot, temperature 0, top
10 by static equity, with a random opening of mean 2 plies. K6 AC.TA is hasty
rank 62. A self-play mover under that cut cannot make a play like it, so
post-setup positions enter the corpus only through the uniform random-opening
plies, which are almost never deliberate setups. The model is asked to value a
position class it has essentially no game outcomes for, and off-distribution
it falls back to "7 points, weak leave, hooks open".

Two facts argue against capacity: the model binds the opponent's F to the FA
hook and the premium (the same rack x cross-check x premium conjunction it
fails on for its own side), and the synthetic cross-check binding tests passed
([film_conditioning_results.md](../film_conditioning_results.md)). A bigger
trunk or more games of the same kind sharpen the on-distribution numbers and
leave the hole intact.

## What the pipeline does today (the facts the plan rests on)

- The teacher trains on `.slog` rows only: one row per sampled eligible turn,
  value targets from the game's final scores, the four placement targets from
  the moves actually played next ([architecture.md](../architecture.md)). No sim
  is in the teacher's loop.
- Monte-Carlo sims exist in two places: the frozen eval sets' ground truth
  (`monte_carlo_sim_tool`), and `.sobs` sidecars for the evidence track
  (`sim_obs_tool` for equity-top-K candidates, `evidence_trajectory_generator`
  for the anchor / on-policy / off-policy recipe). A `.sobs` record per
  candidate carries wins, draws, losses, the delta sum and second moment, and
  the four footprint-class histograms (`SimObservation`, sim_runner.h) --
  exactly the teacher's target set, as distributions instead of one draw.
- Encoding a candidate's post-move state from a replayed `.slog` position is
  what `move_set_eval_target_generator` already does per candidate
  (`encode_post_move_row`), and what `CandidateEvaluator` does at play time.
- Terminal HastyBot rollouts cost about one thread-second per hundred: 1000
  rollouts from an early or mid-game position take 0.7-0.9 s wall on 16
  threads (measured on this box, `monte_carlo_sim_tool`, face-up condition).

## The change

**Phase 1 -- sim targets over all K candidates.** At a sampled subset of
self-play positions, sim K candidates with common random numbers to a natural
end and keep every candidate's post-move state as a teacher training row with
soft targets: WLD from the outcome weights, score-diff mean and predictive
variance from the delta sums, the four placement heads from the footprint
histograms. Rows from the `.slog` games continue as now. The "best" play the
discussion started from is not a selection at all here: the setup play gets
its accurate label as a candidate whether or not anything plays it, and the
sim rows go from one per sampled position to K.

**Phase 2 (gated) -- continue from the sim's pick.** Let the self-play game
proceed from a sampled turn with the sim's pick, sampled at a low temperature
over sim win%, so later positions of the same game descend from sim-quality
play. This needs the sim in-game rather than post hoc, and it perturbs the
game-outcome targets of the positions before it the way random openings do,
so it waits for phase 1's measurements.

**Not doing:** a listwise or pairwise ranking loss over each pool (a rival
design from the plan review). The measured fault is an absolute one located
in a placement head (self-next 0.34 against 0.57), which the histograms
supervise directly, and `NeuralAgent` compares absolute win% across
candidates, so calibrated siblings are already the ranking signal.

**Not doing:** BestBot-vs-BestBot self-play. Simming every move at K
candidates multiplies sim spend by the moves-per-game over
positions-labeled-per-game ratio, best play *after* the labeled turn is
wasted once the sim supplies the targets, and best play *before* it is what
phase 1 subsumes.

## Design

### Candidate set

The equity top-K is the gate on coverage: a rank-62 setup is invisible to
K=10 or 20 under any selection rule. Use the model-free stratified recipe the
`.mset` generator already has (`move_set_eval::stratified_candidates` /
`StratumQuotas`, training/move_set_eval_candidates.h; workload params
`quota_top / quota_mid / quota_tail / quota_exchange`, `mid_rank_limit`): the
head, a sample from the contention zone, a sample from the tail, exchanges.
The tail sample is what reaches setups; the disagreement rate below says how
much of it to buy. The labeler also needs a games-fraction option, which no
sidecar tool has (`--positions-per-game` is an integer applied to every game;
2000 positions from 20,000 games is one game in ten).
The trajectory selector (`evidence_trajectory_select.h`) is a different,
model-dependent recipe and is untouched. `quota_exchange` is 0 for this stream
until the exchange encoding question below is settled.

The selection and the sim loop are a library (`sim/slog_position_simmer.h`)
shared by `sim_obs_tool`, PR 0's `sim_candidate_survey_tool`, and the labeler
below. It reports each candidate's equity rank, which the measurement reads
and the stratum-balanced sibling draw will need; `.sobs` has no field for it,
which is why the measurement is its own tool writing CSV rows rather than new
`sim_obs_tool` flags.

### Rollouts and their condition

Terminal HastyBot rollouts, not value-truncated ones. Terminal sims are
model-independent, so the labels never go stale and need no model-version
stamping (the open question the workload doc defers to "the neural phase"),
and the teacher never trains on its own leaf readouts (the roadmap's stated
constraint for this stream). At about ten thread-milliseconds per rollout
they are affordable at the counts below. Sims run under the corpus's
information condition (face-up for the current teacher).

### Where the labels live: in the `.slog`

The evidence track keeps its sim records in a `.sobs` sidecar because they
depend on a model (the proposer, a truncation leaf) and are regenerated while
the games stay fixed, and because several tags label the same games
differently. Neither holds here: terminal HastyBot labels are as permanent a
fact about a game as its final scores, and there is one recipe. So they are
stored where the final scores are. A new `.slog` version (`kVersion`,
binary_log.h) gains an optional sim-label section, located from the
`FileHeader` and indexed by (game, turn): per labeled turn its candidates, each
a move plus the sparse soft targets (WLD counts, delta sum and second moment,
up to 300 nonzero (class, count) entries per placement head at 300 rollouts).
That is the loader's label format on disk, so nothing is converted at train
time, and it replaces the dense 35,185 B `SimObsRecord` (about 1.1 GB per
generation at K = 16 x 2000 positions) with a few KB per candidate. A file
with no labels has an empty section; the game blobs are unchanged.

What this buys is that a chunk stays one file. The generational scheduler's
one-rename-per-whole-`.slog` invariant, `selfplay_gen.deliver`, the ledger,
quarantine, the bucket mirror and rented-trainer delivery are all untouched,
and a generation's training rows are fixed when it closes. (Rejected: a
same-stem sidecar delivered as a pair, which needs delivery ordering, stem
keying, two-member quarantine and an orphan sweep in a scheduler built for
single files; and a separate labeling role over committed generations, which
adds a role and a trainer whose rows depend on when it looked.)

The version bump makes existing corpora unreadable, as any does. The labeling
tool below is also the converter: it reads an old-version `.slog` and writes a
new-version one, labeled or not.

### Generation

One labeler, two callers:

- **Post hoc, for the offline experiment (PR 1):** a tool that rewrites an
  existing tag's `.slog` files into labeled new-version ones, skipping files
  already at the new version so an interrupted run resumes.
- **In the generator (PR 3, built only after the offline experiment is
  positive):** `play_game` sims the sampled turns of each finished game
  in-process before the writer flushes it, so one process produces the whole
  chunk in one go. Phase 2 needs the sim in the game loop anyway; this puts
  it there once.

New task params: `sim_positions_per_game` (fractional), `sim_rollouts`, the
four quotas, `mid_rank_limit`, landing with defaults that turn the stream off;
migrating live tags that should turn it on is its own follow-up PR. Sims
dominate the generator's wall-clock and make generation cadence sim-bound;
`sim_positions_per_game` and the fleet size are the levers, and ssh workers
scale it like any other chunk producer.

### Training: the second row source

The C++ `DataLoader` learns a second row kind, delivered as its own batch
stream with its own row width; game rows keep today's layout and cost. (Rows
are fixed-width from the single `AllTargets` list, and a placement target is
one float class index today; four dense 2927-class distributions would add
about 11.7k floats to every row, game rows included.) For each labeled
position the loader replays to the decision point (as now) and, per selected
candidate, applies the move and encodes the post-move row
(`encode_post_move_row`, the encoder path the `.mset` generator and the agent
share). The sim row's label block: WLD as a 3-vector, score-diff (mean,
variance), each placement head as a sparse list of up to m (class, probability)
pairs from histogram / n, densified on the GPU (the extra class carries the
pass and not-win mass as the hard targets do), plus a row weight. The
trainer's sim-stream losses are soft cross-entropy for WLD and the placement
heads and the Gaussian score-diff loss against the predictive variance; the
game stream's hard-label losses are unchanged.

Pieces of this that are deliverables of their own, each with tests:

- **A footprint-class transpose table.** The loader applies a per-row diagonal
  symmetry, the stored histograms are in the natural frame, and a transposed
  class is not a cell transpose (its slot moves between the H and V blocks,
  footprint.h). Today the code only transposes a `Move` and classifies it, so
  a class permutation is new. The candidate `Move` is transposed before
  `encode_post_move_row` as well.
- **Masks for an unplayed candidate.** `encode_post_move_row` writes inputs
  only; masks come from `TargetList::encode_all` over an `EncodeContext` bound
  to the game's replay. The decoder needs a route that builds the context from
  the candidate's post-move encoder and leave.
- **The soft-target mask rule.** The loss force-keeps the target class in the
  mask (model.py); with soft targets the whole target support is OR-ed into
  the mask, or a rollout footprint the over-approximate mask missed gives
  -inf x p.
- **The sim-row index.** The loader picks game-row turns at train time
  (`EpochConfig`), independently of the turns the sim tool labeled, so sim
  rows get their own index over (file, labeled position, candidate), with the
  per-epoch sibling draw, the row weight and the file-stem held-out split
  inside the shuffle.

Why in the loader and not Python: the row's input is the replay's job (the
replay-reconstruction invariant), and the epoch shuffle across files is what
keeps siblings out of one batch.

### Reuse control

K siblings share almost all their input planes. The July reuse collapse
(40 samples per game memorized WLD; 4 per game is the regime) is the hazard.
Controls, all in the loader's epoch plan:

- **Subsample siblings per epoch**: each pool contributes its anchor (the
  pool's first stored candidate, the equity argmax -- at temperature 0 also the
  played move) plus `siblings_per_epoch` (default 3) drawn fresh each epoch,
  stratum-balanced so a pool's tail candidates are drawn as often as its head
  ones. Every pass sees a different subset: augmentation, not repetition.
  `siblings_per_epoch` = K turns the control off, which is the ablation that
  says whether it is needed. A labeled turn that is also the epoch's game-row
  turn keeps its game row: the anchor's sim row and the hard-outcome row are
  different targets for the same input, and the board weight covers both.
- **Weight per board**: a pool's rows carry weight so its total gradient
  share is near one game row's.
- **Budget by boards**: games per generation are not reduced by K.
- **Watch the collapse's signature**: held-out eval win MAE on the
  Monte-Carlo position sets, and a train/held-out split of the sim rows by
  file stem reporting sim-target soft-CE on both.

The redundancy is in the inputs, not the information: the differences between
siblings are exactly the quantity `NeuralAgent` consumes, so the sibling rows
are a direct learning-to-rank signal on the model's actual job.

### Budget

With K = 16 and 300 rollouts per candidate, one labeled position is 4800
rollouts, about 4 s wall on 16 threads. Labeling 2000 positions per generation
is then about 2.2 machine-hours per generation on one 16-thread box and yields
32,000 sim rows against the 20,000 game rows a generation carries today. That
is the starting point; the fleet scales it, and the disagreement measurement
below sets K. It is also more than the self-play that fills a generation, so
labeling inline makes generation cadence sim-bound (see Generation).

PR 0 at K = 64, 300 rollouts, 300 positions, two replicas is 11.5 million
rollouts: 33 minutes on 28 threads (measured), and 2 MB of CSV.

## Measurements

**Before building (PR 0):** run the sim at K = 64 with the quota strata over a
few hundred sampled positions of the current corpus and report how often, and
by how much, the sim prefers a candidate outside the hasty top 10. That is
what the cut costs, and it prices K and the tail quota. The best of 64 noisy
estimates flatters itself and most candidates lie outside the cut, so the
figures are held out: each position is simmed twice on independent rollouts,
a pick made on one replica is valued on the other.

**PR 0 result** (`py/scripts/sim_candidate_survey.py`; 300 positions of
`transformer-clipped` gen 2548, face-up, the played move plus the rest of the
top 32 in full, 28 tail, 4 exchanges, 300 rollouts x 2 replicas):

| | all 300 | the 220 undecided (played move at 10-90% win) |
|---|---|---|
| sim pick outside the top 10 | 17.7% of picks | 15.0% |
| ...which the held-out replica confirms (gain > 0) | 45.3% | 42.4% |
| held-out win% gained by lifting the cut, per position | +0.10 +/- 0.10 | -0.09 +/- 0.08 |

Picking by spread instead: +0.24 +/- 0.17 points of spread per position. On
the positions HastyBot's own games reach, the top-10 cut costs nothing
measurable: a pick outside it is confirmed at coin-flip rate, i.e. it is
rollout noise. Picks from rank 32 on were 9 of 600, and two positions carry
all of their gain: one pre-endgame position where every top-10 move loses and
a rank-58 move wins 30%, one +3-point rank-35 play. The tail sample covers a
median 6% of a position's tail (median 530 legal moves), so tail winners are
undercounted by roughly that factor -- but even scaled, a uniform tail draw
finds a sim-confirmed tail winner at well under one labeled position in ten.

**The same test aimed at setups** (`--recipe setup`): only positions with a
"high-value setup" play ranked outside the top 10 -- a J/Q/X/Z kept in the
leave and a tile laid beside an empty premium square where it then hooks
(`sim/setup_plays.h`; K6 AC.TA is the defining case) -- simming the top 10
plus every such play, 1000 rollouts x 2 replicas. 16% of eligible turns
qualify; 300 were simmed. The best setup play loses to the best top-10 move
by 10.3 +/- 0.6 win% on average (held out), the sim's pick lands outside the
cut in 5.5% of picks, and lifting the cut gains +0.06 +/- 0.02 win% per
position. In 11 positions (3.7%) the setup was confirmed better (same pick on
both replicas, both held-out gains positive), in 3 of them by more than 2
points on both, in none by more than 5. The strongest are collected in
`positions/NWL23/setup-survey-examples/`.

What this does and does not say. It confirms the diagnosis's premise from the
other side: setups are rare on-distribution, which is why the corpus lacks
them. It does not measure the quantity sim rows would fix, the *model's*
error on tail candidates (ACETA's ten points), since no model is in this
loop. And it prices the uniform tail stratum poorly as a way to find setups:
its labels would overwhelmingly say "as bad as equity says". Whether the model
already knows that is the measurement that decides the tail quota, and it
needs the model's values over these same candidates.

**Acceptance for phase 1** (the arms compare at equal generations and equal
game rows; the offline experiment fine-tunes both arms from the same
checkpoint):

- A held-out sim-labeled slice (file-stem split) on which the model's ranking
  of candidates is scored against the sim's: rank correlation and top-1
  agreement, overall and on the pools whose sim-best lies outside the hasty
  top 10, reported per generation. This is the gate.
- ACETA is a probe, not a gate: a uniform tail draw reaches a rank-62 play at
  one position with probability near quota_tail / 105, so one position can
  pass or fail for reasons unrelated to the mechanism. Report
  `neural_rank_tool` on it: K6 AC.TA's distance from its sim truth (44%) and
  its rank, J6 AC.TA still near 36, and the `ACETA-no-F-leave.gcg` variant's
  shift against the sim's (under a point, where the model moves 4.6 today).
- Held-out eval win MAE on the Monte-Carlo position sets not worse than the
  same recipe without sim rows; the win/placement metrics of the Positions
  tab unchanged or better.
- Match eval against the fixed opponent not worse.

**Gate for phase 2:** phase 1 lands and the ranking metric above plateaus
with the tail quota still finding disagreements, i.e. coverage of the
*positions* reached, not of candidates at them, is what limits.

## PR slicing

The order puts the evidence before the plumbing: labels can be written post
hoc into an existing tag's games, so the loader, the losses and a fine-tune
can be tried with no generator or workload change. The generator step (task
params, an ssh bundle redeploy) is built only on a positive result.

0. **Measurement (done).** The quota selection and sim loop become a shared
   library; `sim_candidate_survey_tool` sims the K = 64 sample and a script
   reports the held-out disagreement figures and the rank distribution of the
   sim's picks. The games-fraction option moves to PR 1 with the labeler that
   needs it.
1. **Format + loader.** The `.slog` version with the sim-label section, the
   post hoc labeling / converting tool, the footprint-class transpose table,
   the post-candidate `EncodeContext` and masks, the sim-row stream and its
   index (sibling subsampling, weights, the held-out split). The version bump
   itself is a mechanical commit apart from the rest.
2. **Trainer + offline experiment.** The soft losses and mask rule, the
   held-out ranking metrics; an existing tag's generations labeled post hoc;
   a fine-tune from the epoch-2500 teacher with and without sim rows, and the
   `siblings_per_epoch` = K ablation. Go / no-go for the rest.
3. **Generation.** `play_game` labels in-process; params default-off.
   Live-tag migration (`migrate_tag_params.py`) is a follow-up PR of its own.
4. **Run.** A tag from the transformer profile with sim rows on, against the
   same profile without; the acceptance checks above.
5. **Phase 2 (optional).** A `play_game` agent that plays HastyBot except at
   its pre-chosen sampled turns, where it sims the quota set (the labels it
   would have written anyway) and picks by a temperature softmax over sim
   win%; the eligible-region rule for the perturbed game-outcome targets.

## Open questions

- **A semantic tail stratum.** If PR 0 shows the uniform tail is too coarse:
  the plays with the highest score the mover's leave can make next turn if
  the opponent passes -- a hook-setup detector from one move generation. Not
  built before that measurement asks for it.

- **Score-diff variance target.** The stored second moment gives a
  predictive variance per candidate; whether the std head trains on it or
  keeps its current loss is a small experiment.
- **Placement extra-class mass.** The sim histograms count footprints of the
  reply; passes and (for the win heads) not-win rollouts land in the extra
  class exactly as the hard targets define it, but the mapping should be
  checked against `footprint.h` before the loss is written.
- **TensorRT precision on this export.** The BF16 engine prices K6 AC.TA
  about 1.8 points below CPU FP32 (other candidates differ by under 0.3),
  and the FP32 engine build fails on an attention slice node of the
  transformer export. Separate issue; noted so the acceptance numbers are read
  from one path.
