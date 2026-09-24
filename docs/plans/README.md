# Plans and design proposals

Each document here records a decision: the problem, the design chosen, the
alternatives it beat, and how it lands. Each opens with its status. The
system-as-built documents indexed in [../README.md](../README.md) describe
what exists; a plan says why it was built that way. A plan whose work has
landed keeps its rationale and review record even where the code has since
moved on. New plans go here, not in `docs/` proper.

## Modeling and training

- **[sim_residual_feedback.md](sim_residual_feedback.md)**: *built, waiting
  on training.* Feeding Monte Carlo rollout evidence back into the value
  models, re-evaluating the move set conditioned on it, and choosing the next
  candidate to sim with a learned expected-gain ("proves-best") head. The
  kill-test passed; the fusion stage, the move proposal model and the
  sequential agent (UltimateBot) exist. The gen-1 frozen-backbone trial is
  the recorded floor.
- **[rack_conditional_evidence.md](rack_conditional_evidence.md)**:
  *proposed, plan-reviewed, not built.* Extends the sim-residual loop so that
  what the sims of one candidate find transfers to the rest of the turn:
  evidence kept per rack index, an evidence-conditioned reply policy inside
  rollouts, outdated rollouts re-priced or re-run rather than discarded, and
  one acquisition rule over (candidate, rack indices). Built one layer at a
  time against an expert-labeled evaluation set.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)**: *proposed;
  the measurement (PR 0) landed.* A second target stream for the teacher: sim
  outcomes over every simmed candidate at sampled self-play positions, stored
  in the `.slog`, with soft targets and sibling subsampling. Motivated by the
  ACETA setup-play blind spot in `positions/NWL23/interesting-positions/`;
  records the survey of how often plays outside HastyBot's top 10 win.
- **[footprint_native_placement.md](footprint_native_placement.md)**:
  *landed.* Placement made footprint-categorical end to end, with the
  per-cell collapse kept only for visualization: the (15,15,13) spatial
  reshape, per-format storage decisions, the PR slicing and its deviations,
  and the plan-review dissent it resolved.
- **[generational_teacher.md](generational_teacher.md)**: *proposed,
  deferred.* The move-set student tracking an improving teacher: the teacher
  as versioned per-tag state advanced by promotion, teacher-bound corpus
  generations on a pair-aware ingest protocol, and the student training over
  a sliding window.
- **[pov_calibration_bias.md](pov_calibration_bias.md)**: *diagnosis
  settled, fix not built.* The teacher's POV calibration bias (about +0.8%
  win probability / +2.5 points toward the POV player): the evidence chain,
  its decomposition into unpinned drift over a stable, slightly negative
  structure, the candidate fixes and gates, and reproduction recipes.
- **[fp16_safe_serving.md](fp16_safe_serving.md)**: *landed.* The FP16
  activation-overflow incident, resolved by serving BF16: the measured
  monotone activation growth, the BF16/FP16/FP32 comparison, and why the
  model-side containment program was removed.
- **[lexical_features_for_value.md](lexical_features_for_value.md)**:
  *partly built, not adopted.* Lexical foresight for the value models through
  engineered GADDAG-computed input features instead of lexicon knowledge
  inside the network. The contingent-draw potential map was built and
  deleted; the post-move cross-check delta has an encoder and a diagnostic
  but is not yet a model input.

## Cloud compute

- **[cloud_training.md](cloud_training.md)**: *landed, partly superseded.*
  Running trainers on rented GPUs, several tags in parallel: the trainer's
  record/controls contract, the torch runtime and the bucket legs (all still
  in use), and a Runpod train slot, since replaced by cloud_machines.md.
- **[tag_queue.md](tag_queue.md)**: *proposed, not reviewed.* A machine
  pool and an ordered tag queue: each free pool machine takes the next
  eligible queued tag, with slots made from the workload's layout and
  eligibility checked against GPU memory, and releases it when every slot
  has finished. Makes an end condition mandatory for every workload.
- **[cloud_machines.md](cloud_machines.md)**: *landed.* AWS instances as
  task-scoped ssh machines that the dashboard launches, idles and terminates;
  the trainer on the ssh kind; spot; Runpod's removal. The provider lives in
  `py/cloud/providers/`.
