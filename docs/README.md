# Scribblez documentation map

## Start here

- **[design.md](design.md)** — the north-star design: why existing engines are
  beatable (context-blind leave evaluation, naive rack inference), and the
  target architecture — public belief system, unified Q/V network, GADDAG +
  Monte Carlo search.
- **[roadmap.md](roadmap.md)** — the implementation plan: the variant we
  develop in (face-up leaves), the agent everything converges on (one-pass
  candidate scoring, then a sequential sim loop driven by a proves-best head
  over evidence), what is already built, what is left to write, and the three
  models that have to be trained to feed it.
- **[evaluation_plan.md](evaluation_plan.md)** — the other half: what past
  measurements established, the match/eval machinery and its known gaps, and
  the evaluation to run once the agent is built. Deliberately separate, because
  nothing in the roadmap is gated on a result.

## The system as built

- **[architecture.md](architecture.md)** — how a self-play game becomes a
  training row: the component chain, the `.slog` format, the
  replay-reconstruction invariant, random openings.
- **[model_architectures.md](model_architectures.md)** — wiring diagrams for the
  two trained networks and the spatial trunk they share: layer-by-layer shapes,
  the head fan-out of each, and their loss tables.
- **[generational_training.md](generational_training.md)** — the generate→train
  lifecycle (rows-clock, sliding window, reuse-driven epochs, live controls),
  plus the forward-looking game-pool producer, resource-contention manager, and
  distributed-worker design it grows into.
- **[wsd_lr_schedule.md](wsd_lr_schedule.md)** — the trainers' cyclic
  warmup-stable-decay learning-rate schedule: why cycles rather than a single
  decay or a manual control, the exact piecewise shape, the controller's
  bookkeeping, per-trainer sizing, known limitations, and the design-review
  record.
- **[endgame_bench_results.md](endgame_bench_results.md)** — the endgame
  solver's measured cost/strength curve vs its node budget (methodology, the
  seat-mirrored head-to-head protocol, the shipped default of 400).
- **[move_set_eval_results.md](move_set_eval_results.md)** — the A3 curves:
  how well the distilled candidate filter reproduces the teacher's ranking,
  measured on a full-sweep held-out slice against the incumbent static-equity
  shortlist, with what the numbers do and do not establish.
- **[move_set_eval_v2_results.md](move_set_eval_v2_results.md)** — the
  roadmap item 1 close-out: the first `.mset` v2 (planar) corpus and the
  student trained with the placement-plane readouts, against the v1 curves
  and the incumbent.
- **[film_conditioning_results.md](film_conditioning_results.md)** — the
  `use_film` post-mortem: FiLM makes the leave↔cross-check binding expressible
  (and the gate engages), but does not close the pos-09 M7 gap; why (learned
  frequency prior, placement objective at ~1% of the trunk gradient), with
  BatchNorm noise ruled out, and the two experiments that follow.
- **[pov_calibration_bias.md](pov_calibration_bias.md)** — the teacher's
  measured POV calibration bias (+0.8% win-prob / +2.6 pts toward the POV
  player): the evidence chain, its decomposition into a structural
  score-diff under-correction plus a training-drifting offset, and the
  phased fix plan with reproduction recipes and acceptance criteria.
- **[react_dashboard.md](react_dashboard.md)** — the training dashboard: React
  shell + Python data API, embedded Bokeh metric figures, and the interactive
  lane-analysis and Positions tabs.
- **[cloud_compute.md](cloud_compute.md)** — distributed data generation on
  rented cloud CPUs (Runpod): the stable dependency-only worker image, per-arch
  code bundles through R2, the fleet CLI, and results syncing back to the
  local mount for unchanged analysis.
- **[master_dashboard.md](master_dashboard.md)** — the React dashboard as the
  single entrypoint for all work: the workload registry, the job-control
  plane (launch/stop local and cloud workers from the browser), and
  workload-specific analysis tabs.
- **[position_eval_workload.md](position_eval_workload.md)** — the training
  workloads on the master dashboard: the workload-spec contract (roles, params,
  stats, tabs), distributed self-play generation via staging + controller-side
  ingest, and the trainer as a singleton consumer worker.

## Design proposals
- **[fp16_safe_serving.md](fp16_safe_serving.md)** — the FP16 activation-overflow
  incident and its **resolution: serve BF16.** Records the measured monotone
  activation growth, why the model-side containment program (magnitude
  penalties + export gate + FP32 pins) was tried and then retired, and the
  bf16-vs-fp16-vs-fp32 measurement that justified switching the serving format
  instead.
- **[generational_teacher.md](generational_teacher.md)** — AlphaZero-style
  teacher broadcast for the distillation pipeline: the teacher as versioned
  per-tag state advanced by one-click (later automatic) promotion,
  teacher-bound corpus generations on a pair-aware ingest protocol, and the
  student training over a sliding window.
- **[sim_residual_feedback.md](sim_residual_feedback.md)** — feeding Monte-Carlo
  rollout evidence back into the value models for evidence-conditioned
  re-evaluation, and picking the next candidate to sim via a learned
  expected-gain (proves-best) head. Steps 1–4 of its implementation roadmap
  are done (the kill-test passed; the fusion stage is built), and the gen-1
  frozen trial is recorded as the floor the move proposal model replaces.
- **[sim_obs_experiment_results.md](sim_obs_experiment_results.md)** — the
  kill-test's numbers, controls, and conclusions.
- **[lexical_features_for_value.md](lexical_features_for_value.md)** — giving
  the value models lexical foresight through engineered GADDAG-computed input
  features (the contingent-draw potential map, the cross-check delta) instead
  of network-internal lexical knowledge. The potential map was built and has
  since been removed; the doc keeps the rationale.

## The lexical-NN experiment track

Probe experiments asking whether a network can internalize or query the
lexicon. Their findings — a compiled-lexicon tool helps only when its structure
matches the task's shape, and a plain network cannot spell a word's interior —
are what justify the engineered-feature approach above.

- **[lexical_tools.md](lexical_tools.md)** — the catalog of compiled-lexicon
  modules (DAWG walks, anagram search, KV memory) and the registry interface.
- **[word_validity_experiments.md](word_validity_experiments.md)** — experiment
  1: classifying real words vs. statistically-matched phonies; the ordered DAWG
  walk wins.
- **[rack_best_experiments.md](rack_best_experiments.md)** — experiment 2: the
  best word in an unordered rack; the anagram tool wins where the ordered walk
  fails, and generation is genuinely harder than discrimination.
- **[lexical_nn.md](lexical_nn.md)** — experiment 3: the max-move-per-lane
  board task (per-lane best-move union + score heads on a CNN + lane
  transformer).
