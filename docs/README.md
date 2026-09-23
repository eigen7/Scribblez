# Scribblez documentation

The index of everything under `docs/`. Documents fall into four kinds: the
direction (why and what next), the system as built (how it works today),
results (what a run or experiment measured), and plans (the record of a
decision, in [plans/](plans/README.md)).

## Direction

- **[design.md](design.md)**: the north-star design. Why existing engines are
  beatable (context-blind leave evaluation, naive rack inference), and the
  target architecture: a public belief system, a unified Q/V network, GADDAG
  move generation and Monte Carlo search.
- **[roadmap.md](roadmap.md)**: the implementation plan. The variant
  development runs in (face-up leaves), the agent everything converges on
  (one-pass candidate scoring, then a sequential sim loop driven by a
  proves-best head over evidence), the status of each item, and the three
  models that feed it.
- **[evaluation_plan.md](evaluation_plan.md)**: the measurement half. What past
  measurements established, the match and evaluation machinery and its known
  gaps, and the evaluation to run once the agent is built. Kept separate
  because nothing in the roadmap is gated on a result.

## The system as built

Data and models:

- **[architecture.md](architecture.md)**: how a self-play game becomes a
  training row. The component chain, the `.slog` format, the
  replay-reconstruction invariant, random openings, seeding.
- **[model_architectures.md](model_architectures.md)**: wiring diagrams and
  loss tables for the two trained networks and the spatial trunk they share,
  including the evidence fusion stage, the proves-best head, and the
  evidence-path ONNX split.

Training:

- **[generational_training.md](generational_training.md)**: the generate→train
  lifecycle (rows-clock, sliding window, reuse-driven epochs, live controls),
  plus the forward-looking game-pool producer, resource-contention manager and
  distributed-worker design it grows into.
- **[wsd_lr_schedule.md](wsd_lr_schedule.md)**: the trainers' cyclic
  warmup-stable-decay learning-rate schedule. Why cycles rather than a single
  decay or a manual control, the exact shape, the controller's bookkeeping,
  per-trainer sizing, known limitations, and the design-review record.
- **[position_eval_workload.md](position_eval_workload.md)**: the training
  workloads on the master dashboard. The workload-spec contract (roles,
  params, stats, tabs), distributed self-play generation via staging and
  controller-side ingest, and the trainer as a singleton consumer worker.

Dashboard and compute:

- **[master_dashboard.md](master_dashboard.md)**: the React dashboard as the
  single entrypoint for all work. The workload registry, the job-control plane
  (launching and stopping local, ssh and rented workers from the browser), and
  workload-specific analysis tabs.
- **[react_dashboard.md](react_dashboard.md)**: the dashboard's
  implementation. A React shell over a Python data API, embedded Bokeh metric
  figures, and the interactive Positions, Trajectories and Lane analysis tabs.
- **[cloud_compute.md](cloud_compute.md)**: how work runs on machines
  Scribblez does not own. The dependency-only worker image, per-arch code
  bundles, and the bucket that brings results back to the local mount so
  analysis runs unchanged.
- **[blind_spots.md](blind_spots.md)**: collecting positions where a play from
  outside HastyBot's top moves out-sims all of them. Running the `blind_spots`
  workload across a fleet, what workers deliver, and turning a tag into a
  committed examples directory.

## Results

- **[endgame_bench_results.md](endgame_bench_results.md)**: the endgame
  solver's measured cost and strength against its node budget. Methodology,
  the seat-mirrored head-to-head protocol, and the shipped default of 400.
  The captured margin-sweep run behind its figures is
  [data/endgame_margin_sweep.txt](data/endgame_margin_sweep.txt).
- **[move_set_eval_results.md](move_set_eval_results.md)**: the v1 (A3) curves.
  How well the distilled candidate filter reproduces the teacher's ranking on
  a full-sweep held-out slice, against the static-equity shortlist, and what
  the numbers do and do not establish.
- **[move_set_eval_v2_results.md](move_set_eval_v2_results.md)**: the run that
  closed roadmap item 1. The first corpus with per-candidate placement planes
  and the student trained on them, against the v1 curves and the static-equity
  shortlist.
- **[sim_obs_experiment_results.md](sim_obs_experiment_results.md)**: the
  sim-evidence kill-test. Its numbers, controls and verdict (pass).
- **[film_conditioning_results.md](film_conditioning_results.md)**: the
  `use_film` post-mortem. FiLM makes the leave-to-cross-check binding
  expressible, and the gate engages, but it does not close the gap on the
  motivating test position. Why (a learned frequency prior; a placement
  objective at ~1% of the trunk gradient), with BatchNorm noise ruled out, and
  the two experiments that follow.
- **[pos6-analysis.txt](pos6-analysis.txt)**: a captured sim-evidence probe
  session on test position `pos-6`: per-candidate sim evidence and the
  opponent-reply hot spots. The probe script it came from no longer exists.

## Analyses

- **[analysis/richards-johnson-exchange/](analysis/richards-johnson-exchange/README.md)**:
  Nigel Richards's EELLT exchange from the 2025 World Cup, which no engine
  setting approves of, and why it may nonetheless be the winning play. Board
  images are rendered by the repo's own board renderer; the directory holds
  the game, the variants and the render config.

## Plans and design proposals

Plan-reviewed designs, incident write-ups with their fix plans, and the
proposals behind landed or deferred work. A plan is the record of a decision;
the documents above describe the system as it is.
[plans/README.md](plans/README.md) says more about each, including whether it
has landed.

- **[plans/sim_residual_feedback.md](plans/sim_residual_feedback.md)**: feeding
  rollout evidence back into the value models, and choosing the next candidate
  to sim with a learned expected-gain (proves-best) head.
- **[plans/rack_conditional_evidence.md](plans/rack_conditional_evidence.md)**:
  keeping evidence per sampled opponent rack so knowledge found simming one
  candidate transfers to the rest of the turn.
- **[plans/sim_labeled_candidates.md](plans/sim_labeled_candidates.md)**: the
  teacher's second target stream, sim outcomes over every simmed candidate at
  sampled self-play positions.
- **[plans/generational_teacher.md](plans/generational_teacher.md)**: advancing
  the teacher by promotion, with teacher-bound corpus generations and a
  student trained over a sliding window.
- **[plans/footprint_native_placement.md](plans/footprint_native_placement.md)**:
  making placement footprint-categorical end to end.
- **[plans/pov_calibration_bias.md](plans/pov_calibration_bias.md)**: the
  teacher's measured bias toward the point-of-view player, its diagnosis, and
  the phased fix plan.
- **[plans/fp16_safe_serving.md](plans/fp16_safe_serving.md)**: the FP16
  activation-overflow incident and its resolution, serving in BF16.
- **[plans/lexical_features_for_value.md](plans/lexical_features_for_value.md)**:
  giving the value models lexical foresight through engineered, GADDAG-computed
  input features instead of network-internal lexical knowledge.
- **[plans/cloud_training.md](plans/cloud_training.md)**: running the trainers
  on rented GPUs, several tags in parallel.
- **[plans/cloud_machines.md](plans/cloud_machines.md)**: rented AWS instances
  as task-scoped ssh machines the dashboard launches, idles and terminates.

## The lexical-NN experiment track

Probe experiments asking whether a network can internalize or query the
lexicon. Their findings (a compiled-lexicon tool helps only when its structure
matches the task's shape, and a plain network cannot spell a word's interior)
are what justify giving the value models engineered lexical features instead
([plans/lexical_features_for_value.md](plans/lexical_features_for_value.md)).

- **[lexical_tools.md](lexical_tools.md)**: the catalog of compiled-lexicon
  modules (DAWG walks, anagram search, KV memory) and the registry interface.
- **[word_validity_experiments.md](word_validity_experiments.md)**: experiment
  1, classifying real words against statistically matched phonies. The ordered
  DAWG walk wins.
- **[rack_best_experiments.md](rack_best_experiments.md)**: experiment 2, the
  best word in an unordered rack. The anagram tool wins where the ordered walk
  fails, and generation is genuinely harder than discrimination.
- **[lexical_nn.md](lexical_nn.md)**: experiment 3, the max-move-per-lane board
  task (per-lane best-move and score heads on a CNN plus a lane transformer).

## Images

[images/](images/) holds the figures the documents embed. The `arch_*.svg`
architecture diagrams are generated by
[py/tools/plot_model_architectures.py](../py/tools/plot_model_architectures.py)
and the `endgame_*.svg` figures by
[py/tools/plot_endgame_bench.py](../py/tools/plot_endgame_bench.py); edit the
scripts, not the SVGs.
