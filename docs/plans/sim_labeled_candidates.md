# Sim-labeled candidate rows for the position evaluation model

**Status: proposed; only the measurement (PR 0) has landed.** PR 0's shared
sim library and survey tool are in the tree
(`sim/slog_position_simmer.h`, `sim_candidate_survey_tool`,
`py/scripts/sim_candidate_survey.py`), along with work that grew out of it:
the exhaustive recipe, the survey viewer, solved endgames in the confirming
sims, and the `blind_spots` dashboard workload that collects outside-the-cut
winners across machines ([blind_spots.md](../blind_spots.md)).
`neural_rank_tool --sim` prints the sim beside the model's ranking. Nothing
from PR 1 onward exists: no sim-label section in the `.slog`, no sim-row
stream in the loader, no soft-target losses.

**Problem.** The position evaluation model has a measured blind spot for
setup plays, and the self-play corpus cannot teach it: HastyBot's top-10
equity cut never plays them, so their outcomes never reach training.

**Decision.** Give the teacher a second target stream. At a sampled subset of
self-play positions, sim K candidates to the end of the game and keep every
candidate's post-move state as a training row with soft targets. A later,
gated phase lets the sim's pick steer the self-play game past the sampled
turn. The roadmap carries the first half as the teacher's "planned second
target stream" ([roadmap.md](../roadmap.md), *The position evaluation
model*); this is the concrete route to it. Why the search levels above
BestBot's need sibling-accurate values is argued in
[simulation_levels.md](../simulation_levels.md).

## The finding

`positions/NWL23/interesting-positions/ACETA.gcg`, turn 2: Sokol to move with
AACITTZ, after Anderson's GREEK kept EF. Sokol plays K6 AC.TA for 7 points.
The two A's land to the right of the triple-letter squares J6 and J10, so ZA
scores 31 at either, and the C between them means no two-letter word ending
in C can block both. The `transformer-clipped` teacher (epoch 2500) ranks the
play 101st of 137 legal placements.

Twelve candidates (the model's top ten plus both ACETA variants) were simmed
to the end of the game, 2000 HastyBot rollouts each, under both information
conditions (`monte_carlo_sim_tool`, standard error about 1.1 points):

| play | model win% | sim face-up | sim hidden |
|---|---|---|---|
| 9K TIZ | 44.0 | 44.1 | 43.3 |
| K6 AC.TA | 33.9 | 44.1 | 43.0 |
| L3 ATTAC. | 40.8 | 41.1 | 40.8 |
| J6 AC.TA (the A's *on* the TLs) | 36.8 | 36.4 | 36.2 |

Every on-distribution candidate is priced within a point of the sim; the
setup misses by ten. This is a specific blind spot, not a calibration offset.

The placement heads show where. After K6 AC.TA, the model's opponent-next
plane is right (J10-J12 hot, the FA hook, matching the sim's 41-55%
coverage), but its self-next plane puts 0.34 on the mover's Z hook where the
sim's rollouts put 0.57: it sees the threat and undercounts its own payoff.
And the F in the opponent's leave moves the sim not at all (44.1 → 44.2 with
the F replaced by an O) but moves the model by 4.6 points. The model treats
the two hooks as independent threats, where the sim, like the human, sees
that only one of them can be taken.

Reproduce the readouts with

```
neural_rank_tool --gcg positions/NWL23/interesting-positions/ACETA.gcg --turn 2 -k 0 --model <onnx>
```

(the committed file records the whole game, so `--turn 2` selects the
position; K6 is the played move). Add `--sim` for the sim win rate and rank
of every scored move beside the model's.

## Diagnosis: coverage, not capacity

The tag generated its games HastyBot against HastyBot, at temperature 0,
picking from the top 10 by static equity, with random openings of mean 2
plies. K6 AC.TA is HastyBot's rank 62. A self-play mover under that cut cannot
make such a play, so post-setup positions enter the corpus only through the
uniformly random opening plies, which are almost never deliberate setups. The
model is asked to value a class of positions it has essentially no game
outcomes for, and off-distribution it falls back to "7 points, weak leave,
hooks open".

Two facts argue against a capacity limit. The model does bind the opponent's
F to the FA hook and the premium square, which is the same rack × cross-check
× premium conjunction it fails on for its own side. And the synthetic
cross-check binding tests pass
([film_conditioning_results.md](../film_conditioning_results.md)). A bigger
trunk, or more games of the same kind, would sharpen the on-distribution
numbers and leave the hole intact.

## What the pipeline does today

- The teacher trains on `.slog` rows only: one row per sampled eligible turn,
  value targets from the game's final scores, and the four placement targets
  from the moves actually played next ([architecture.md](../architecture.md)).
  No sim is in the teacher's loop.
- Monte-Carlo sims exist in two places: the frozen eval sets' ground truth
  (`monte_carlo_sim_tool`), and the evidence track's `.sobs` sidecars
  (`sim_obs_tool` for equity-top-K candidates, `evidence_trajectory_generator`
  for the anchor / on-policy / off-policy recipe). A `.sobs` record per
  candidate carries wins, draws, losses, the delta sum and second moment, and
  the four footprint-class histograms (`SimObservation`, `sim_runner.h`):
  exactly the teacher's target set, as distributions instead of one draw.
- Encoding a candidate's post-move state from a replayed `.slog` position is
  what `move_set_eval_target_generator` already does per candidate
  (`encode_post_move_row`), and what `CandidateEvaluator` does at play time.
- Terminal HastyBot rollouts cost about one thread-second per hundred: 1000
  rollouts from an early or mid-game position take 0.7 to 0.9 s wall on 16
  threads (measured with `monte_carlo_sim_tool`, face-up condition).

## The change

**Phase 1: sim targets over all K candidates.** At a sampled subset of
self-play positions, sim K candidates with common random numbers to the end
of the game, and keep every candidate's post-move state as a teacher training
row with soft targets: WLD from the outcome counts, score-diff mean and
predictive variance from the delta sums, and the four placement heads from
the footprint histograms. Rows from the `.slog` games continue as now. Nothing
is "selected" here: the setup play gets an accurate label as a candidate
whether or not anything plays it, and sim rows go from one per sampled
position to K.

**Phase 2 (gated): continue from the sim's pick.** Let the self-play game
proceed from a sampled turn with the sim's pick, sampled at low temperature
over sim win%, so later positions in the same game descend from sim-quality
play. This needs the sim inside the game rather than after it, and it
perturbs the game-outcome targets of earlier positions the way random
openings do, so it waits for phase 1's measurements.

**Not doing: a listwise or pairwise ranking loss** over each pool (a rival
design from the plan review). The measured fault is an absolute error in a
placement head (self-next 0.34 against 0.57), which the histograms supervise
directly, and `NeuralAgent` compares absolute win% across candidates, so
calibrated siblings already are the ranking signal.

**Not doing: BestBot-vs-BestBot self-play.** Simming every move at K
candidates multiplies sim spend by the ratio of moves per game to labeled
positions per game. Best play *after* the labeled turn is wasted once the sim
supplies the targets, and best play *before* it is what phase 1 subsumes.

## Design

### Candidate set

The equity top-K is the coverage gate: a rank-62 setup is invisible to K=10
or 20 under any selection rule. Use the model-free stratified recipe the
`.mset` generator already has (`move_set_eval::stratified_candidates` /
`StratumQuotas` in `training/move_set_eval_candidates.h`; workload params
`quota_top`, `quota_mid`, `quota_tail`, `quota_exchange`, `mid_rank_limit`):
the head, a sample from the contention zone, a sample from the tail, and
exchanges. The tail sample is what reaches setups, and the disagreement rate
below says how much of it to buy. `quota_exchange` is 0 for this stream until
it is settled how an exchange candidate's row is encoded. The model-dependent
trajectory selector (`training/evidence_trajectory_select.h`) is a different
recipe and is untouched.

The labeler also needs an option to label a fraction of games, which no
sidecar tool has: `--positions-per-game` is an integer applied to every game,
and 2000 positions from 20,000 games is one game in ten.

Selection and the sim loop are a library (`sim/slog_position_simmer.h`)
shared by `sim_obs_tool`, PR 0's `sim_candidate_survey_tool`, and the labeler
below. It reports each candidate's equity rank, which the measurement reads
and a stratum-balanced sibling draw will need. `.sobs` has no field for the
rank, which is why the measurement is its own tool writing CSV rather than
new `sim_obs_tool` flags.

### Rollouts and their information condition

Terminal HastyBot rollouts, not value-truncated ones. Terminal sims are
model-independent, so the labels never go stale and need no model-version
stamp, and the teacher never trains on its own leaf readouts (the roadmap's
constraint on this stream). At about ten thread-milliseconds per rollout they
are affordable at the counts below. Sims run under the corpus's information
condition (face-up for the current teacher).

### Where the labels live: in the `.slog`

The evidence track keeps its sim records in a `.sobs` sidecar for two
reasons: they depend on a model (the proposer, a truncation leaf) and are
regenerated while the games stay fixed, and several tags label the same games
differently. Neither holds here. Terminal HastyBot labels are as permanent a
fact about a game as its final scores, and there is one recipe, so they are
stored where the final scores are.

A new `.slog` version (`kVersion` in `binary_log.h`) gains an optional
sim-label section, located from the `FileHeader` and indexed by (game, turn).
Per labeled turn it holds the candidates, each a move plus its sparse soft
targets: WLD counts, the delta sum and second moment, and up to 300 nonzero
(class, count) entries per placement head at 300 rollouts. That is the
loader's label format on disk, so nothing is converted at train time. It
replaces the dense 35,185 B `SimObsRecord` (about 1.1 GB per generation at
K = 16 × 2000 positions) with a few KB per candidate. A file with no labels
has an empty section, and the game blobs are unchanged.

What this buys is that a chunk stays one file. The generational scheduler's
one-rename-per-`.slog` invariant, the generator's delivery path, the ledger,
quarantine, the bucket mirror and remote-trainer delivery are all untouched,
and a generation's training rows are fixed when it closes. Rejected
alternatives: a same-stem sidecar delivered as a pair, which needs delivery
ordering, stem keying, two-member quarantine and an orphan sweep in a
scheduler built for single files; and a separate labeling role over committed
generations, which adds a role and makes the trainer's rows depend on when
that role got to them.

The version bump makes existing corpora unreadable, as any bump does. The
labeling tool below is also the converter: it reads an old-version `.slog`
and writes a new-version one, labeled or not.

### Generation

One labeler, two callers:

- **Post hoc, for the offline experiment (PR 1):** a tool that rewrites an
  existing tag's `.slog` files into labeled new-version ones, skipping files
  already at the new version so an interrupted run resumes.
- **In the generator (PR 3, built only if the offline experiment is
  positive):** `play_game` sims the sampled turns of each finished game
  in-process before the writer flushes it, so one process produces the whole
  chunk. Phase 2 needs the sim in the game loop anyway; this puts it there
  once.

New task params: `sim_positions_per_game` (fractional), `sim_rollouts`, the
four quotas, and `mid_rank_limit`, landing with defaults that turn the stream
off; migrating the live tags that should turn it on is a follow-up PR. Sims
dominate the generator's wall-clock and make generation cadence sim-bound;
`sim_positions_per_game` and the fleet size are the levers, and ssh workers
scale it like any other chunk producer.

### Training: a second row source

The C++ `DataLoader` learns a second kind of row, delivered as its own batch
stream with its own row width, while game rows keep today's layout and cost.
(Rows are fixed-width from the single `AllTargets` list, and a placement
target is one float class index today; four dense 2927-class distributions
would add about 11.7k floats to every row, game rows included.)

For each labeled position the loader replays to the decision point as now
and, per selected candidate, applies the move and encodes the post-move row
with `encode_post_move_row`, the encoder path the `.mset` generator and the
agent share. The sim row's label block: WLD as a 3-vector; score-diff mean
and variance; each placement head as a sparse list of up to m (class,
probability) pairs from histogram / n, densified on the GPU (the extra class
carries the pass and not-win mass, as the hard targets do); and a row weight.
The trainer's sim-stream losses are soft cross-entropy for WLD and the
placement heads, and the Gaussian score-diff loss against the predictive
variance. The game stream's hard-label losses are unchanged.

Pieces of this that are deliverables of their own, each with tests:

- **A footprint-class transpose table.** The loader applies a diagonal
  symmetry per row, the stored histograms are in the natural frame, and a
  transposed class is not a cell transpose: its slot moves between the H and
  V blocks (`footprint.h`). Today the code only transposes a `Move` and
  classifies it, so a class permutation is new. The candidate `Move` is
  transposed before `encode_post_move_row` as well.
- **Masks for an unplayed candidate.** `encode_post_move_row` writes inputs
  only; masks come from `TargetList::encode_all` over an `EncodeContext` bound
  to the game's replay. The decoder needs a route that builds that context
  from the candidate's post-move encoder and leave.
- **The soft-target mask rule.** The loss force-keeps the target class in
  the legality mask (`position_eval/model.py`). With soft targets, the whole
  target support must be OR-ed into the mask, or a rollout footprint that the
  over-approximate mask missed yields −inf × p.
- **The sim-row index.** The loader picks game-row turns at train time
  (`EpochConfig`), independently of the turns the labeler chose, so sim rows
  get their own index over (file, labeled position, candidate), with the
  per-epoch sibling draw, the row weight, and the held-out split by file stem
  inside the shuffle.

Why the loader and not Python: a row's input is the replay's job (the
replay-reconstruction invariant), and the epoch shuffle across files is what
keeps siblings out of one batch.

### Reuse control

K siblings share almost all of their input planes. The July reuse collapse
(40 samples per game memorized WLD; 4 per game is the working regime) is the
hazard. The controls, all in the loader's epoch plan:

- **Subsample siblings per epoch.** Each pool contributes its anchor (its
  first stored candidate, the equity argmax, which at temperature 0 is also
  the played move) plus `siblings_per_epoch` (default 3) drawn fresh each
  epoch, stratum-balanced so a pool's tail candidates are drawn as often as
  its head ones. Every pass sees a different subset: augmentation, not
  repetition. Setting `siblings_per_epoch` = K turns the control off, which
  is the ablation that says whether it is needed. A labeled turn that is also
  the epoch's game-row turn keeps its game row: the anchor's sim row and the
  hard-outcome row are different targets for the same input, and the board
  weight covers both.
- **Weight per board**: a pool's rows carry weights so its total gradient
  share is about one game row's.
- **Budget by boards**: games per generation are not reduced by K.
- **Watch for the collapse's signature**: held-out eval win MAE on the
  Monte-Carlo position sets, and a train/held-out split of the sim rows by
  file stem, reporting sim-target soft-CE on both.

The redundancy is in the inputs, not the information: the differences
between siblings are exactly what `NeuralAgent` consumes, so the sibling rows
are a direct learning-to-rank signal on the model's actual job.

### Budget

With K = 16 and 300 rollouts per candidate, one labeled position is 4800
rollouts, about 4 s wall on 16 threads. Labeling 2000 positions per
generation is then about 2.2 machine-hours per generation on one 16-thread
box, and yields 32,000 sim rows against the 20,000 game rows a generation
carries today. That is the starting point; the fleet scales it, and the
disagreement measurement below sets K. It is also more compute than the
self-play that fills a generation, so labeling inline makes generation cadence
sim-bound (see Generation).

PR 0 at K = 64, 300 rollouts, 300 positions and two replicas is 11.5 million
rollouts: 33 minutes on 28 threads (measured), and 2 MB of CSV.

## Measurements

**What PR 0 asks.** Sim K = 64 candidates with the quota strata over a few
hundred sampled positions of the current corpus, and report how often, and by
how much, the sim prefers a candidate outside HastyBot's top 10. That is what
the cut costs, and it prices K and the tail quota. The best of 64 noisy
estimates flatters itself, and most candidates lie outside the cut, so the
figures are held out: each position is simmed twice on independent
rollouts, and a pick made on one replica is valued on the other.

**Endgame caveat for all three results below.** They were measured with
greedy rollout endgames. The survey later found greedy endgames misjudging
late-game candidates by tens of win%, and its confirming sims now solve
endgames once at most 14 tiles are unseen (commit `ba55503`). The pre-endgame
rows of the exhaustive table are the most exposed. The example positions
measured that way (`positions/NWL23/sim-survey-examples/`) were removed; the
`blind_spots` workload's output replaces them.

**PR 0 result** (`py/scripts/sim_candidate_survey.py`; 300 positions of
`transformer-clipped` generation 2548, face-up; the played move plus the rest
of the top 32 in full, 28 from the tail, 4 exchanges; 300 rollouts × 2
replicas):

| | all 300 | the 220 undecided (played move at 10-90% win) |
|---|---|---|
| sim pick outside the top 10 | 17.7% of picks | 15.0% |
| ...confirmed by the held-out replica (gain > 0) | 45.3% | 42.4% |
| held-out win% gained by lifting the cut, per position | +0.10 ± 0.10 | −0.09 ± 0.08 |

Picking by spread instead: +0.24 ± 0.17 points of spread per position. On the
positions HastyBot's own games reach, the top-10 cut costs nothing
measurable: a pick outside it is confirmed at a coin-flip rate, which is
rollout noise. Picks from rank 32 on were 9 of 600, and two positions carry
all of their gain: a pre-endgame position where every top-10 move loses and a
rank-58 move wins 30%, and a +3-point rank-35 play. The tail sample covers a
median 6% of a position's tail (median 530 legal moves), so tail winners are
undercounted by about that factor. Even scaled up, a uniform tail draw finds
a sim-confirmed tail winner at well under one labeled position in ten.

**The same test aimed at setups** (`--recipe setup`). Only positions with a
"high-value setup" play ranked outside the top 10: a J, Q, X or Z kept in the
leave, and a tile laid beside an empty premium square where that tile then
hooks (`sim/setup_plays.h`; K6 AC.TA is the defining case). Sim the top 10
plus every such play, 1000 rollouts × 2 replicas. 16% of eligible turns
qualify; 300 were simmed. The best setup play loses to the best top-10 move
by 10.3 ± 0.6 win% on average (held out); the sim's pick lands outside the
cut in 5.5% of picks; lifting the cut gains +0.06 ± 0.02 win% per position.
In 11 positions (3.7%) the setup was confirmed better (the same pick on both
replicas, both held-out gains positive), in 3 of them by more than 2 points
on both, in none by more than 5. The strongest are in
`positions/NWL23/setup-survey-examples/`.

**The exhaustive version** (the survey's default recipe). At 900 random
eligible positions from 3000 fresh HastyBot games, every legal play that
places no blank was screened at 1000 rollouts, racing (a candidate three
paired standard errors below the leader stops early). The screen's five best
plays from outside the top 10 were then re-simmed beside the top 10 at 5000
fresh rollouts. A play counts when that confirming sim puts it at least two
paired standard errors above the best top-10 move. 133 plays at 54 positions
(6.0%) do; playing the best of them would gain +0.47 ± 0.12 win% per
position. By bag size at the decision:

| bag | positions | with a confirmed outside play | mean gain there | cost per position |
|---|---|---|---|---|
| 51+ | 368 | 1 (0.3%) | 2.0 | 0.01 |
| 21-50 | 326 | 10 (3.1%) | 2.1 | 0.07 |
| 8-20 | 132 | 20 (15.2%) | 2.9 | 0.44 |
| 1-7 | 74 | 23 (31.1%) | 14.7 | 4.56 |

As measured, the cut's cost is a pre-endgame phenomenon: bag-timing and
who-goes-out decisions that static equity cannot see, often worth tens of
win%, often with many equivalent plays (every one-tile play that leaves one
tile in the bag). With more than 20 tiles in the bag a confirmed outside play
is rare and worth about 2 win%. (This is where the endgame caveat above bites
hardest.) The survey's `.simsurvey.json` output carries per-candidate
statistics (margin histogram, both sides' next-move scores, the end-of-game
rack settlement) for a classifier of why such plays win.

**What this does and does not say.** It confirms the diagnosis's premise
from the other side: setups are rare on-distribution, which is why the corpus
lacks them. It does not measure what sim rows would fix, the *model's* error
on tail candidates (ACETA's ten points), since no model is in this loop. And
it shows the uniform tail stratum to be a poor way to find setups: its labels
would overwhelmingly say "as bad as equity says". Whether the model already
knows that is the measurement that decides the tail quota, and it needs the
model's values over these same candidates.

**Acceptance for phase 1.** The arms compare at equal generations and equal
game rows; the offline experiment fine-tunes both arms from the same
checkpoint.

- **The gate:** a held-out sim-labeled slice (split by file stem) on which
  the model's ranking of candidates is scored against the sim's, by rank
  correlation and top-1 agreement, overall and on the pools whose sim-best
  lies outside the HastyBot top 10, reported per generation.
- **ACETA is a probe, not a gate.** A uniform tail draw reaches a rank-62
  play at one position with probability near `quota_tail` / 105, so a single
  position can pass or fail for reasons unrelated to the mechanism. Report
  `neural_rank_tool` on it: K6 AC.TA's distance from its sim truth (44%) and
  its rank; J6 AC.TA still near 36; and the shift of the
  `ACETA-no-F-leave.gcg` variant against the sim's (under a point, where the
  model moves 4.6 today).
- Held-out eval win MAE on the Monte-Carlo position sets no worse than the
  same recipe without sim rows; the Positions tab's win and placement metrics
  unchanged or better.
- Match eval against the fixed opponent no worse.

**Gate for phase 2:** phase 1 lands, and the ranking metric above plateaus
while the tail quota is still finding disagreements. That would mean coverage
of the *positions* reached, not of the candidates at them, is what limits.

## PR slicing

Evidence before plumbing: labels can be written post hoc into an existing
tag's games, so the loader, the losses and a fine-tune can be tried with no
generator or workload change. The generator step (task params, an ssh bundle
redeploy) is built only on a positive result.

0. **Measurement. Landed.** The quota selection and sim loop became a shared
   library; `sim_candidate_survey_tool` sims the K = 64 sample, and a script
   reports the held-out disagreement figures and the rank distribution of the
   sim's picks. The games-fraction option moved to PR 1, with the labeler
   that needs it.
1. **Format and loader.** The `.slog` version with the sim-label section; the
   post hoc labeling/converting tool; the footprint-class transpose table; the
   post-candidate `EncodeContext` and masks; the sim-row stream and its index
   (sibling subsampling, weights, the held-out split). The version bump itself
   is a mechanical commit apart from the rest.
2. **Trainer and offline experiment.** The soft losses and the mask rule; the
   held-out ranking metrics; an existing tag's generations labeled post hoc;
   a fine-tune from the epoch-2500 teacher with and without sim rows, and the
   `siblings_per_epoch` = K ablation. Go / no-go for the rest.
3. **Generation.** `play_game` labels in-process; params default off.
   Live-tag migration (`migrate_tag_params.py`) is a follow-up PR of its own.
4. **Run.** A tag from the transformer profile with sim rows on, against the
   same profile without, under the acceptance checks above.
5. **Phase 2 (optional).** A `play_game` agent that plays HastyBot except at
   its pre-chosen sampled turns, where it sims the quota set (the labels it
   would write anyway) and picks by a temperature softmax over sim win%; plus
   the rule for which game-outcome targets the perturbation leaves eligible.

## Open questions

- **A semantic tail stratum.** PR 0 found the uniform tail a poor way to
  reach setups. The alternative: the plays whose leave can score the most
  next turn if the opponent passes, a hook-setup detector from one move
  generation. Not built until the model-side measurement above asks for it.
- **Score-diff variance target.** The stored second moment gives a predictive
  variance per candidate; whether the std head trains on it or keeps its
  current loss is a small experiment.
- **Placement extra-class mass.** The sim histograms count the reply's
  footprints; passes and (for the win heads) not-win rollouts land in the
  extra class exactly as the hard targets define it, but check the mapping
  against `footprint.h` before writing the loss.
- **TensorRT precision on the transformer export.** The BF16 engine prices K6
  AC.TA about 1.8 points below CPU FP32 (other candidates differ by under
  0.3), and the FP32 engine build fails on an attention slice node of the
  transformer export. A separate issue, noted so the acceptance numbers are
  all read from one path.
