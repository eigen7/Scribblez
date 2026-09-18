# Plans and design proposals

Each document here records a decision: the problem, the design chosen, the
alternatives it beat, and how it lands. The system-as-built documents in
[../README.md](../README.md) describe what exists; a plan describes why it was
built that way, and a plan whose work has landed keeps its review record and
rationale even where the code has since moved on. New plans go here, not in
`docs/` proper.

## Modeling and training

- **[sim_labeled_candidates.md](sim_labeled_candidates.md)** — plan
  for the teacher's second target stream: sim outcomes over every simmed
  candidate at sampled self-play positions (K post-move rows per position,
  soft targets, sibling subsampling), motivated by the ACETA setup-play blind
  spot in `positions/NWL23/interesting-positions/`.
- **[sim_residual_feedback.md](sim_residual_feedback.md)** — feeding Monte-Carlo
  rollout evidence back into the value models for evidence-conditioned
  re-evaluation, and picking the next candidate to sim via a learned
  expected-gain (proves-best) head. Steps 1–4 of its implementation roadmap
  are done (the kill-test passed; the fusion stage is built), and the gen-1
  frozen trial is recorded as the floor the move proposal model replaces.
- **[generational_teacher.md](generational_teacher.md)** — AlphaZero-style
  teacher broadcast for the distillation pipeline: the teacher as versioned
  per-tag state advanced by one-click (later automatic) promotion,
  teacher-bound corpus generations on a pair-aware ingest protocol, and the
  student training over a sliding window.
- **[footprint_native_placement.md](footprint_native_placement.md)** — the plan to
  make placement footprint-categorical end to end (removing the per-cell collapse
  outside visualization): the (15,15,13) spatial reshape, sparse top-k storage,
  the PR slicing, and the plan-review dissent it resolved.
- **[pov_calibration_bias.md](pov_calibration_bias.md)** — the teacher's
  measured POV calibration bias (+0.8% win-prob / +2.6 pts toward the POV
  player): the evidence chain, its decomposition into a structural
  score-diff under-correction plus a training-drifting offset, and the
  phased fix plan with reproduction recipes and acceptance criteria.
- **[fp16_safe_serving.md](fp16_safe_serving.md)** — the FP16 activation-overflow
  incident and its **resolution: serve BF16.** Records the measured monotone
  activation growth, why the model-side containment program (magnitude
  penalties + export gate + FP32 pins) was tried and then retired, and the
  bf16-vs-fp16-vs-fp32 measurement that justified switching the serving format
  instead.
- **[lexical_features_for_value.md](lexical_features_for_value.md)** — giving
  the value models lexical foresight through engineered GADDAG-computed input
  features (the contingent-draw potential map, the cross-check delta) instead
  of network-internal lexical knowledge. The potential map was built and has
  since been removed; the doc keeps the rationale.

## Cloud compute

- **[cloud_training.md](cloud_training.md)** — plan-reviewed design
  for running the trainers on rented GPU pods, several tags in parallel: the
  trainer's record/controls contract (landed), the runtime (landed), and the
  Runpod cloud slot (landed, since removed).
- **[cloud_machines.md](cloud_machines.md)** — plan-reviewed design
  for rented machines: AWS instances as task-scoped ssh machines the
  dashboard launches, idles and terminates; the trainer on the ssh kind;
  Runpod's removal. The provider lives in `py/cloud/providers/`.
