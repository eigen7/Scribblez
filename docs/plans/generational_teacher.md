# Generational teacher broadcast

**Status: proposed, deferred, not built.** Today the `move_set_eval` workload
distills from one frozen teacher per tag: `teacher_tag` plus
`teacher_generation`, resolved to a concrete position_eval export when the
task is created. Its pairs go to a flat `slogs/` pair store, and its trainer
makes repeated passes over that growing store
(`py/scribblez/workloads/move_set_eval.py`). Nothing below exists yet.

**Goal.** Let the move_set_eval student track an improving position
evaluation model instead of a teacher frozen per tag.

**Decision.** Make the teacher versioned per-tag state, advanced by an
explicit **promotion**. Each generation of the corpus binds to exactly one
teacher, and the student trains over a sliding window of generations that
ages superseded targets out. The implementation plan is at the end.

## Motivation

Under the frozen-teacher design ([roadmap.md](../roadmap.md), tracks A2 and
A3), refreshing the teacher means a new tag and a full corpus regeneration.
That is the wrong shape for the loop this project is building toward. The
position evaluation model improves as its own training advances, and the
student should follow it: an AlphaZero-style cycle in which a better position
evaluation model periodically becomes the new teacher and produces a better
student, which in turn produces better position evaluation training data once
self-play uses the student-based agent.

This design closes only the **inner** loop: position_eval → teacher →
student. position_eval trains on HastyBot self-play, a fixed policy, so the
student does not yet feed back into its own data. The outer loop closes with
neural self-play ([generational_training.md](../generational_training.md)
steps 3 and 4). The broadcast machinery here (content-addressed model
artifacts, a versioned pointer, digest-stamped outputs) is the "model
distribution to generators" that step 4 calls for, built once and reused
there.

Two facts make a teacher refresh cheap enough to do routinely:

- The engine's TensorRT plan cache is keyed on the ONNX's architecture
  signature, weights excluded. On a cache hit the engine *refits* the cached
  plan with the new checkpoint's weights
  ([neural_net.cpp](../../engine/src/nn/neural_net.cpp)), which is far
  cheaper than an engine build. Only an architecture change forces a cold
  build.
- A move_set_eval cycle runs the target generator as a subprocess that takes
  `--model`. Nothing holds the model across cycles, so swapping teachers at a
  cycle boundary means passing a different path.

## The teacher record

The frozen `teacher_tag` / `teacher_generation` params give way to a
mutable per-tag **teacher record**, `teacher.json` under the tag root, owned
by the controller and written atomically:

```json
{"epoch": 3,
 "sha256": "<digest>",
 "path": "<tag root>/teachers/<sha256>/model_epoch_0007.onnx",
 "object_key": null,
 "source": {"workload": "position_eval", "tag": "<src>", "generation": 7}}
```

- `epoch` counts promotions within the tag and only increases.
- The model bytes are **copied, content-addressed, into the student tag's
  own tree** (`teachers/<sha256>/<basename>`). Local resolution then never
  depends on another tag's lifetime (the source position_eval tag can be
  deleted without leaving this tag's teachers dangling), and every teacher
  version stays resolvable for as long as its generations exist.
- `object_key` is set once the bucket leg exists (below): the same bytes as a
  content-addressed bucket object, so remote workers can fetch them and
  verify the digest.
- Task creation seeds epoch 0 through a workload **creation hook**, a small
  `WorkloadSpec` extension from validated params to side effects. The seed
  params are two plain strings, the source position_eval tag and the
  checkpoint filename; the hook validates them, hashes the ONNX, copies it
  in, and writes the epoch-0 record.

## Promotion

Promotion is one function: validate, ingest the model, bump the record. Its
**caller** is deliberately abstract, because the operator flow comes first
and automation later.

- **v1: one click.** The task's Controls tab gets a promote control. Because
  the record carries the source tag, "promote the source tag's latest export"
  is a single button, showing the checkpoint name, with a filename override
  for promoting something other than the latest. Cadence and gating stay with
  the operator, the same philosophy as the operator-stepped learning rate,
  who reads the source tag's match_eval and eval curves that the dashboard
  already shows.
- **Later: a policy.** Auto-promotion is a controller-side caller of the same
  function, for example a scheduler-tick policy "promote every Nth export
  whose match_eval win rate clears a threshold", with its knobs as params or
  live controls. So nothing in the promotion path may assume a human is
  behind it: validation must be complete, the function must be idempotent
  under retry (the epoch bump guarded by the source checkpoint's identity),
  and every promotion must be recorded in an event log, so plots can mark
  teacher changes whether a click or a policy caused them.

Promotion **validates two compatibility axes** before accepting a checkpoint.
Weights and architecture may change freely, but two things may not:
`opp_leave_input` (the ONNX metadata input arm the engine and FFI session
adopt) and the tag's information condition (`face_up_leaves`). The FFI input
arm is process-wide and fixed per run, the student's input layout was fixed
at task creation, and `MsetDataset` refuses mixed condition flags within a
tag. That check stays per tag; only the teacher-hash check becomes per
generation.

Promotion **does not touch generation directories**. It writes the record.
The generation scheduler, which stays the single writer of generation
structure, notices on its next tick that the record's epoch is ahead of the
open generation's and acts:

- If the open generation has no pairs, its manifest is **restamped** with the
  new teacher and it stays open. This also makes promoting immediately after
  task creation well-defined.
- Otherwise the open generation is sealed complete at its committed count,
  and the next generation opens under the new teacher. A generation sealed
  below a `min_pairs` threshold is flagged in its manifest, so metric plots
  can mark or skip it instead of rendering noise.

## Teacher-bound generations

move_set_eval adopts the generational structure of
[generational_training.md](../generational_training.md) (staging delivery,
`data/generations/gen_NNNNNN/` with manifests, scheduler assignment, trainer
cursor pacing) in place of the flat `slogs/` pair store.

- Each generation's manifest records `{teacher_epoch, teacher_sha256}` when it
  opens or is restamped. The `.mset` single-teacher invariant becomes
  **per generation**: `MsetDataset` requires exactly one teacher hash per
  directory, matching its manifest; hashes may differ across a window's
  directories.
- On assignment, the scheduler checks the staged pair's `.mset` teacher hash
  against the open generation's. A stale pair must be discarded *correctly*:
  its stem is written to the ledger, then its bucket objects are moved to a
  `discarded/` prefix through the mirror machinery. Otherwise the cloud-sync
  watcher re-downloads the pair every interval and the discard loops forever.
  Discards are logged in the scheduler's existing style, with no new UI. They
  are bounded: workers read the record at cycle boundaries, so each worker
  loses at most one in-flight cycle per promotion.
- **Retention: move_set_eval generations are never evicted.** The
  position_eval trainer deletes generations that fall out of its window
  because they are cheap, regenerable CPU self-play. A `.mset` pair embodies
  GPU labeling by the teacher and, in a local-only setup, has no bucket copy.
  The training window slides; aged-out directories stay on disk. Revisit only
  under real disk pressure, once a bucket copy exists.

### The pair-ingest protocol

The generation scheduler's ingest protocol handles single files: its crash
safety rests on one `.slog` moved by one atomic rename, with the ledger line
written first. move_set_eval's unit is a two-file `.slog` / `.mset` pair, so
the protocol generalizes to a **stem** (the pair's shared basename), as a
per-workload ingest strategy on the scheduler. The single-file path stays the
default for other workloads.

- A pair is discoverable in staging only once its `.mset` is present. This
  tolerates cloud sync copying files in any order: a `.slog` on its own is
  pending, not an orphan. (Uploads go `.mset` first, but the puller does not
  preserve that order.)
- Assignment: write the stem to the ledger, rename the `.slog`, then rename
  the `.mset` last. A generation directory counts a pair when its `.mset` is
  present, matching the pair store's own delivery convention, so a crash
  between the renames leaves a pending pair that can be completed, never a
  half-counted one.
- The scheduler's `mirror` hook moves both bucket objects of a stem;
  quarantine (`.bad`) applies to the whole stem; committed counts are
  recomputed each tick by counting `.mset`s that have a companion `.slog`.
- Crash-window tests are part of the deliverable: kill the process between
  the ledger write and the first rename, and between the two renames, and
  assert that the next tick heals it.

## Workers

At the start of each cycle the generate worker reads the teacher record: the
tag file for a local worker, an `rclone cat` of the mirrored record (one
small GET per cycle) for a remote one. When the digest has changed it
resolves the model (a local path, or a digest-verified fetch of the
content-addressed bucket object) and runs the cycle with the new `--model`.
A refresh with an unchanged architecture costs a TensorRT refit on the next
generator run; an architecture change costs one cold plan build per worker.
The worker records the teacher epoch it used in its provenance record; the
`.mset` already carries the hash.

## Student training

The student trainer becomes the workload's singleton train role on the
generational lifecycle: the rows-clock, the train-state cursor, one epoch per
completed generation over the last `W` generations, restart reconciliation,
and live learning-rate control. It does not delete evicted generations
(above). A window spanning teachers is accepted by design: targets from a
superseded teacher are slightly stale but correlated with the current ones,
and the window ages them out. This is the argument of
generational_training.md's "why a window, not a wipe". `W=1` recovers
wipe-per-teacher if staleness proves harmful.

The gate metrics (A3's top-K recall and teacher-value regret) need held-out
data labeled by each generation's *own* teacher; a frozen per-tag split goes
stale at the first promotion. So the scheduler diverts every Nth assigned
pair into the generation's `holdout/` subdirectory, stamped with that
generation's teacher and excluded from the training window. Per-generation
metrics run there, and the curves stay well-defined across promotions.

## The bucket leg

Remote generators already have the substrate this needs
([cloud_compute.md](../cloud_compute.md)): the `-torch` worker image, and
`RoleSpec.inputs`, which stages a role's out-of-tag files where its slot will
look. move_set_eval's generate role uses it today to ship its pinned teacher
to rented GPU machines through the bucket.

What promotion adds on top: uploading the promoted model as a
content-addressed bucket object, and mirroring `teacher.json` to a
controller-maintained prefix outside cloud sync's pull set, so workers can
poll it. The upload must not run on the dashboard's single-threaded IOLoop,
so it runs in an executor or subprocess, bounded by promotion cadence. None
of this is needed for local-only operation, which is where the design proves
out first.

## Alternative considered: teacher-epoch lanes

A rival decomposition uses immutable teacher-epoch directories as the
ingestion partition: workers deliver into per-epoch staging lanes keyed by
the hash every `.mset` already carries, and the trainer advances on its own
committed-data cursor. This fully decouples promotion cadence from optimizer
cadence and never discards a late pair.

It was set aside because it has to rebuild, as new replay-window machinery,
the trainer-side accounting the generational lifecycle already gets right
(the reuse bound, ahead-gating, restart reconciliation). Its two advantages
shrink to near zero once zero-pair restamping and correct bounded discard
exist, and both designs need the pair-ingest protocol anyway. Revisit if
promotion cadence ever needs to decouple from training cadence.

## Implementation plan

**Slice A: the mechanism, proved entirely locally.**

1. The pair-ingest protocol on the generation scheduler, with crash-window
   tests. Independent of anything teacher-related; lands first.
2. Generational move_set_eval: staging delivery through the pair-store loop;
   teacher-stamped manifests; sealing and restamping on promotion via the
   scheduler tick; hash-mismatch discard; holdout diversion; retention; the
   per-generation hash rule in `MsetDataset`; the student trainer on the
   lifecycle.
3. Teacher record and promotion: the record schema and atomic writes; the
   `WorkloadSpec` creation hook and seeding; the promotion function with
   input-arm and condition validation; the one-click promote on the Controls
   tab; the promotion event log; the teacher epoch in worker provenance.
4. Local milestone, the acceptance test for Slice A, on the dev machine with
   local workers only: seed, generate, promote mid-run, generate under the
   new teacher, train across a window spanning both. Verify that epoch N and
   N+1 pairs land in distinct generations, that a stale pair is discarded
   exactly once, the zero-pair restamp path, and per-generation metrics
   against each generation's own teacher. This alone proves the design.

**Slice B: remote reach (independent of Slice A).**

5. The promotion bucket leg: the off-IOLoop upload, `teacher.json`
   mirroring, and the per-cycle poll on remote workers. Its smoke test only
   confirms that a remote worker picks up a rotated teacher; lifecycle
   correctness is already proved by item 4.

**Deliberately out of scope:** neural self-play (the outer loop);
re-labeling old `.slog`s under a new teacher (the same GPU cost as fresh
targets, over a frozen state distribution); the auto-promotion policy itself
(designed for, not built); per-teacher segmentation of throughput plots.

## Open questions

- **Promotion cadence against the window.** Operator judgment under manual
  promotion. The default expectation: promote at exports that pass
  match_eval, with `W=4` as in position_eval. Once the POV-bias coherence
  gate exists ([pov_calibration_bias.md](pov_calibration_bias.md)), it
  belongs in promotion validation.
- **Whether teacher staleness across a window hurts the student.** Unknown
  until measured; `W=1` is the escape hatch.
- **The holdout diversion rate.** Pick N so per-generation metric noise is
  acceptable at the default generation size; measure at the local milestone.
