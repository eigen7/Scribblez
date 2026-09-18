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
position; the K6 line is the played move).

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

**Not doing:** BestBot-vs-BestBot self-play. Simming every move at K
candidates multiplies sim spend by the moves-per-game over
positions-labeled-per-game ratio, best play *after* the labeled turn is
wasted once the sim supplies the targets, and best play *before* it is what
phase 1 subsumes.

## Design

### Candidate set

The equity top-K is the gate on coverage: a rank-62 setup is invisible to
K=10 or 20 under any selection rule. Use the stratified recipe the trajectory
selector already defines (`evidence_trajectory_select.h`, workload params
`quota_top / quota_mid / quota_tail / quota_exchange`, `mid_rank_limit`): the
head, a sample from the contention zone, a sample from the tail, exchanges.
The tail sample is what reaches setups; the disagreement rate below says how
much of it to buy. `sim_obs_tool` (the generation-0, no-model tool) grows the
quota selection; the trajectory generator keeps its proposer-driven recipe.
A cheap semantic stratum to add later if the uniform tail is too coarse: the
plays with the highest score the mover's leave can make next turn if the
opponent passes -- a hook-setup detector from one move generation.

### Rollouts and their condition

Terminal HastyBot rollouts, not value-truncated ones. Terminal sims are
model-independent, so the sidecars never go stale and need no model-version
stamping (the open question the workload doc defers to "the neural phase"),
and the teacher never trains on its own leaf readouts (the roadmap's stated
constraint for this stream). At about ten thread-milliseconds per rollout
they are affordable at the counts below. Sims run under the corpus's
information condition (face-up for the current teacher).

### Generation

Extend the position_eval workload's `generate` role the way
`evidence_trajectories` already composes its cycle: `play_game` writes the
chunk, then `sim_obs_tool` labels a sampled subset of its positions into a
same-stem `.sobs`, and the pair is delivered together (`pair_store` semantics:
a chunk is a `.slog` plus its sidecar). The scheduler ingests pairs into the
generation directory; a `.slog` without its sidecar is not complete. New task
params: `sim_positions_per_game`, `sim_rollouts`, the four quotas,
`mid_rank_limit`. Sims dominate the cycle's wall-clock, so the generate role's
thread count sizes them; ssh workers scale it like any other chunk producer.

### Training: the second row source

The C++ `DataLoader` learns a second row kind. For each labeled position it
replays to the decision point (as now) and, per selected candidate, applies
the move and encodes the post-move row (`encode_post_move_row`, the encoder
path the `.mset` generator and the agent share). The label block gains a
soft-target variant: WLD as a 3-vector, score-diff (mean, variance), each
placement head as a distribution over footprint classes (histogram / n; the
extra class carries the pass and not-win mass as the hard targets do), plus a
row weight. The trainer's losses become soft cross-entropy for WLD and the
placement heads and the existing Gaussian score-diff loss against the
predictive variance; a hard target is the one-hot special case, so one loss
serves both kinds.

Why in the loader and not Python: the row's input is the replay's job (the
replay-reconstruction invariant), and the epoch shuffle across files is what
keeps siblings out of one batch.

### Reuse control

K siblings share almost all their input planes. The July reuse collapse
(40 samples per game memorized WLD; 4 per game is the regime) is the hazard.
Controls, all in the loader's epoch plan:

- **Subsample siblings per epoch**: each pool contributes the anchor
  (the played or equity-argmax move) plus `siblings_per_epoch` (default 3)
  drawn fresh each epoch. Every pass sees a different subset: augmentation,
  not repetition.
- **Weight per board**: a pool's rows carry weight so its total gradient
  share is near one game row's.
- **Budget by boards**: games per generation are not reduced by K.
- **Watch the collapse's signature**: held-out eval win MAE on the
  Monte-Carlo position sets, and a train/held-out split of the sim rows by
  file stem (as the pair store splits) reporting sim-target soft-CE on both.

The redundancy is in the inputs, not the information: the differences between
siblings are exactly the quantity `NeuralAgent` consumes, so the sibling rows
are a direct learning-to-rank signal on the model's actual job.

### Budget

With K = 16 and 300 rollouts per candidate, one labeled position is 4800
rollouts, about 4 s wall on 16 threads. Labeling 2000 positions per generation
is then about 2.2 machine-hours per generation on one 16-thread box and yields
32,000 sim rows against the 20,000 game rows a generation carries today. That
is the starting point; the fleet scales it, and the disagreement measurement
below sets K.

## Measurements

**Before building (PR 0):** run the sim at K = 64 with the quota strata over a
few hundred sampled positions of the current corpus and report the fraction
where the sim's best candidate lies outside the hasty top 10. That is the
share of positions where the cut costs something, and it prices K and the
tail quota.

**Acceptance for phase 1:**

- `neural_rank_tool` on the ACETA position: K6 AC.TA within 2 points of its
  sim truth (44%) and in the top three; J6 AC.TA still near 36. The
  `ACETA-no-F-leave.gcg` variant should move the model by about what it moves
  the sim (under a point), not by 4.6.
- Held-out eval win MAE on the Monte-Carlo position sets not worse than the
  same recipe without sim rows; the win/placement metrics of the Positions
  tab unchanged or better.
- A held-out sim-labeled slice (file-stem split) on which the model's ranking
  of candidates is scored against the sim's: rank correlation and top-1
  agreement, reported per generation. This is the general form of the ACETA
  check.
- Match eval against the fixed opponent not worse.

**Gate for phase 2:** phase 1 lands and the ranking metric above plateaus
with the tail quota still finding disagreements, i.e. coverage of the
*positions* reached, not of candidates at them, is what limits.

## PR slicing

0. **Measurement.** `sim_obs_tool` gains the quota strata; a script runs it at
   K = 64 over sampled positions and reports the disagreement rate and the
   rank distribution of the sim's best. Also lands the ACETA reproduction as
   a documented recipe.
1. **Generation.** The position_eval generate role composes `play_game` +
   `sim_obs_tool`, delivers pairs, the scheduler ingests pairs; new task
   params with a migration for live tags (`migrate_tag_params.py`).
2. **Loader + trainer.** The sim row kind in `DataLoader` / `BlockDecoder`,
   the soft label block, the losses, sibling subsampling and weights, the
   held-out sim slice and its metrics.
3. **Run.** A tag from the transformer profile with sim rows on, against the
   same profile without; the acceptance checks above.
4. **Phase 2 (optional).** A `play_game` agent that plays HastyBot except at
   its pre-chosen sampled turns, where it sims the quota set and picks by a
   temperature softmax over sim win%, writing the `.sobs` record inline; the
   eligible-region rule for the perturbed game-outcome targets.

## Open questions

- **Exchanges as sim rows.** `encode_post_move_row` reduces the rack by the
  move's glyphs, which is right for placements; a post-exchange row needs the
  exchanged tiles removed instead. Either fix the encoder for exchanges or
  keep the exchange stratum labels-only until it is.
- **Score-diff variance target.** The sidecar's second moment gives a
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
