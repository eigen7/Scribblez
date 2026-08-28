# Generational teacher broadcast

How the move_set_eval workload tracks an improving position-eval model
instead of being pinned to one frozen teacher per tag: the teacher becomes
versioned per-tag state advanced by an explicit **promotion**, each
generation of the corpus binds to exactly one teacher, and the student
trains over a sliding window that ages superseded targets out. Design
proposal — not yet built; the implementation plan is at the end.

## Motivation

The distillation pipeline ([roadmap.md](roadmap.md) A2/A3) freezes its teacher
(the `teacher_tag`/`teacher_generation` params) per tag, and the roadmap's
original stance was that refreshing the teacher means a new tag and a full
corpus regeneration. That
is the wrong shape for the loop this project is building toward: the
position-eval model improves as its own training run advances, and the
student should track it — an AlphaZero-style cycle where a better
position-eval periodically becomes the new teacher, producing a better
student, which in turn (once self-play uses the student-based agent)
produces better position-eval training data.

Stated honestly, this design closes the **inner** loop only: position_eval
→ teacher → student. position_eval today trains on HastyBot self-play — a
fixed policy — so the student does not yet feed back into its data. The
outer arc closes with neural self-play
([generational_training.md](generational_training.md) steps 3–4, roadmap
A4/D2). The broadcast machinery here — content-addressed model artifacts, a
versioned pointer, digest-stamped outputs — is exactly the "model
distribution to generators" that step 4 names, built once and reused there.

Two facts make teacher refresh cheap enough to do routinely:

- The engine's TRT plan cache is keyed on the ONNX's architecture signature
  (weights excluded); on a cache hit the engine *refits* the cached plan
  with the new checkpoint's weights
  ([neural_net.cpp](../engine/src/nn/neural_net.cpp)) — far cheaper than an
  engine build. Only an architecture change forces a cold build.
- A move_set_eval cycle invokes the target generator as a subprocess taking
  `--model`; nothing holds the model across cycles, so swapping teachers at
  a cycle boundary is just passing a different path.

## The teacher record

The frozen teacher (the `teacher_tag`/`teacher_generation` params) is replaced
by a per-tag mutable **teacher record**, `teacher.json` under the tag root,
controller-owned, written atomically:

```json
{"epoch": 3,
 "sha256": "<digest>",
 "path": "<tag root>/teachers/<sha256>/model_epoch_0007.onnx",
 "object_key": null,
 "source": {"workload": "position_eval", "tag": "<src>", "generation": 7}}
```

- `epoch` counts promotions within the tag, monotonically.
- The model bytes are **copied, content-addressed, into the student tag's
  own tree** (`teachers/<sha256>/<basename>`): local resolution never
  crosses tag lifetimes (the source position_eval tag can be deleted
  without dangling this tag's teachers), and every teacher version stays
  resolvable for as long as its generations exist.
- `object_key` is set when the cloud leg exists (below): the same bytes as
  a content-addressed bucket object, so pods can fetch digest-verified.
- Task creation seeds epoch 0 through a workload **creation hook** (a small
  WorkloadSpec extension: validated params → side effects). The seed params
  are two ordinary strings — source position_eval tag + checkpoint
  filename — validated by the hook, which hashes the ONNX, copies it in,
  and writes the epoch-0 record.

## Promotion

Promotion is one function — validate, ingest the model, bump the record —
with the **caller** deliberately abstracted, because the operator flow
comes first and automation comes later:

- **v1: one click.** The task's Control tab gets a promote control:
  because the record carries the source tag, "promote the source tag's
  latest export" is a single button (with the checkpoint name shown, and a
  manual filename override for promoting something other than the latest).
  Cadence and gating judgment stay with the operator — the same
  philosophy as the operator-stepped learning rate — reading the source
  tag's match_eval and eval curves that the dashboard already shows.
- **Later: a policy.** Auto-promotion is a controller-side caller of the
  same function — e.g. a scheduler-tick policy "promote every Nth export
  whose match_eval win rate clears a threshold," with its knobs as params or
  live controls. Nothing in the promotion path may assume a human is
  behind it: validation must be complete (no "the operator surely
  checked"), the function idempotent under retry (epoch bump guarded by
  the source checkpoint identity), and every promotion recorded
  (rows-clock-style event log) so plots can annotate teacher changes
  whether a click or a policy caused them.

Promotion **validates both compatibility axes** before accepting a
checkpoint: weights and architecture may change freely, but
`opp_leave_input` (the ONNX metadata arm the engine and FFI session adopt)
and the tag's information condition (`face_up_leaves`) must not — the FFI
input arm is process-wide and immutable per run, the student's input layout
was fixed at task creation, and MsetDataset refuses mixed condition flags
per tag (that check stays per-tag; only the teacher-hash check becomes
per-generation).

Promotion **does not touch generation directories**. It writes the record;
the generation scheduler — still the single writer of generation
structure — observes on its next tick that the record's epoch exceeds the
open generation's and acts:

- open generation has zero pairs → its manifest is **restamped** with the
  new teacher (no seal; this also makes promote-immediately-after-create
  well-defined);
- otherwise → sealed complete at its committed count, and the next
  generation opens under the new teacher. A generation sealed below a
  `min_pairs` threshold is flagged in its manifest so metric plots can
  mark or skip it rather than render noise.

## Teacher-bound generations

move_set_eval adopts the generational structure
([generational_training.md](generational_training.md)): staging delivery,
`data/generations/gen_NNNNNN/` with manifests, scheduler assignment,
trainer cursor pacing — replacing the flat `slogs/` pair store.

- Each generation's manifest records `{teacher_epoch, teacher_sha256}` at
  open (or restamp). The `.mset` single-teacher invariant becomes
  **per-generation**: MsetDataset requires exactly one hash per directory,
  matching its manifest; across a window's directories hashes may differ.
- On assignment the scheduler validates the staged pair's `.mset` teacher
  hash against the open generation's. A stale pair is discarded
  *correctly*: its stem is ledgered, then its bucket objects are moved to a
  `discarded/` prefix through the mirror machinery — otherwise the
  cloud-sync watcher re-downloads the pair every interval and the discard
  loops forever. Discards are logged (the scheduler's existing style, no
  new UI) and bounded: workers poll the record at cycle boundaries, so at
  most one in-flight cycle per worker per promotion is lost.
- **Retention: eviction never deletes move_set_eval generations.** The
  position_eval trainer's evict-beyond-window deletes cheap regenerable
  CPU self-play; a `.mset` pair embodies teacher GPU labeling and, in the
  local-only configuration, has no bucket copy. The training window
  slides; aged-out directories stay on disk. Revisit only under real disk
  pressure, after the bucket mirror exists.

### The pair-ingest protocol

The generation scheduler's ingest protocol is single-file today — its
crash-safety rests on one `.slog` moved by one atomic rename, ledger line
first. move_set_eval's unit is a two-file `.slog`/`.mset` pair, so the
protocol generalizes to a **stem** (the pair's shared basename), as a
per-workload ingest strategy on the scheduler (the single-file path stays
the default for the other workloads):

- A pair is discoverable in staging only when its `.mset` is present. This
  tolerates cloud_sync's arbitrary copy order: a `.slog` alone is pending,
  not an orphan (uploads are mset-first, but the puller does not preserve
  that order).
- Assignment: ledger the stem, then rename `.slog`, then `.mset` last —
  the generation directory's commit point is `.mset` presence, mirroring
  the pair store's own delivery convention, so a crash between renames
  leaves a re-completable pending pair, never a half-counted one.
- `hooks.mirror` moves both bucket objects per stem; quarantine (`.bad`)
  applies to the stem; committed counts are recomputed per tick by
  counting `.mset`s with a companion `.slog`.
- Crash-window tests are part of the deliverable: kill between ledger and
  first rename, and between the two renames; assert self-heal on the next
  tick.

## Workers

At each cycle start the generate worker reads the teacher record (local
workers: the tag file; cloud/ssh workers: an `rclone cat` of the mirrored
record — one small GET per cycle). On a digest change it resolves the model
(local path, or a digest-verified fetch of the content-addressed bucket
object) and runs the cycle with the new `--model`. A same-architecture
refresh costs a TRT refit on the next generator invocation; an
architecture-changing promotion costs one cold plan build per worker. The
worker records the teacher epoch it used in its provenance record; the
`.mset` already carries the hash.

## Student training

The student trainer is the workload's singleton train role on the
generational lifecycle — rows-clock, train-state cursor, one epoch per
completed generation over the last `W` generations, restart
reconciliation, live LR control — minus eviction deletion (above).
Cross-teacher windows are accepted by design: targets from a superseded
teacher are slightly stale but correlated, and the window ages them out —
the same argument as generational_training.md's "why a window, not a
wipe". `W=1` recovers wipe-per-teacher if staleness proves harmful.

Gate metrics (A3's top-K recall and teacher-value regret) need held-out
data labeled by each generation's *own* teacher — a frozen per-tag split
goes stale at the first promotion. The scheduler therefore diverts every
Nth assigned pair into the generation's `holdout/` subdirectory
(teacher-stamped like its generation, excluded from the training window);
per-generation metrics run there, so curves stay well-defined across
promotions.

## The cloud leg

Cloud/ssh workers additionally need the GPU-workloads substrate tracked in
[cloud_compute.md](cloud_compute.md) (the CUDA-capable worker image and
content-addressed artifact fetch). On top of that, promotion uploads the
model as a content-addressed bucket object and mirrors `teacher.json` to a
controller-maintained prefix (outside cloud_sync's pull set). The upload
must not run on the dashboard's single-threaded IOLoop — that process
moves no payload bytes today (mirrors are server-side R2 renames) — so it
runs in a subprocess/executor; a new class of behavior for that process,
bounded by promotion cadence. None of this is needed for local-only
operation, which is where the design proves out first.

## An alternative considered: teacher-epoch lanes

A rival decomposition: immutable teacher-epoch directories as the
ingestion partition (workers deliver into per-epoch staging lanes keyed by
the hash already in every `.mset`), with the trainer advancing on its own
committed-data cursor — fully decoupling promotion cadence from optimizer
cadence and never discarding late pairs. It was set aside because its cost
is rebuilding the trainer-side accounting the generational lifecycle
already gets right (the reuse bound, ahead-gating, restart
reconciliation) as new replay-window machinery, while its two wins shrink
to near-zero once zero-pair restamping and correct bounded discard exist.
Both designs need the pair-ingest protocol either way. Revisit if
promotion cadence ever needs to decouple from training cadence.

## Implementation plan

**Slice A — the mechanism, proved entirely locally.**

1. The pair-ingest protocol on the generation scheduler, with crash-window
   tests. Independent of anything teacher-related; lands first.
2. Generational move_set_eval: staging delivery via the pair-store loop;
   teacher-stamped manifests; promotion sealing/restamping via the
   scheduler tick; hash-mismatch discard; holdout diversion; retention;
   the MsetDataset per-generation hash rule; the student trainer on the
   lifecycle.
3. Teacher record + promotion: the record schema and atomic writes; the
   WorkloadSpec creation hook and seeding; the promotion function with
   arm/condition validation; the Control-tab one-click promote; the
   promotion event log; teacher epoch in worker provenance.
4. Local milestone (Slice A acceptance), on the dev machine with local
   workers only: seed, generate, promote mid-run, generate under the new
   teacher, train across a window spanning both; verify epoch N and N+1
   pairs land in distinct generations, a stale pair is discarded exactly
   once, the zero-pair restamp path, and per-generation metrics against
   each generation's own teacher. This alone proves the design and lets
   the A3 shakeout run generationally on the dev GPU.

**Slice B — cloud reach (independent).**

5. The cloud_compute.md GPU-workloads substrate (CUDA-capable worker
   image; digest-verified artifact fetch), per that document.
6. The teacher-broadcast bucket leg: off-IOLoop promotion upload;
   teacher.json mirroring; worker-side per-cycle poll. Its smoke test only
   confirms a cloud worker fetches a rotated teacher — lifecycle
   correctness is already proved by item 4.

**Deliberately not in scope:** neural self-play (the outer loop);
re-targeting old `.slog`s under a new teacher (same GPU cost as fresh
targets, frozen state distribution); the auto-promotion policy itself
(designed for, not built); per-teacher segmentation of throughput plots.

## Open questions

- Promotion cadence vs window: operator judgment under manual promotion;
  the default expectation is promote at match_eval-passing exports, with
  `W=4` as in position_eval.
- Whether teacher staleness across a window measurably hurts the student:
  unknown until A3 slice-1 numbers exist; `W=1` is the escape hatch.
- The holdout diversion rate: pick N so per-generation metric noise is
  acceptable at the default generation size; measure at the local
  milestone.
