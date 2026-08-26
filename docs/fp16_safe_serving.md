# FP16-safe serving — activation-magnitude control and the pin retirement plan

Proposal: make FP16-safety a **property of the models** — activation
magnitudes bounded in training, enforced by an export/promotion gate — and
retire the runtime's FP32-pinning machinery once the first gated teacher
lands. The engine then serves plain FP16 with no per-layer exceptions, full
FP16 mantissa everywhere, and the fastest kernel path.

The governing principle (per project direction): past runs and
already-exported checkpoints are never a reason to keep accommodation
machinery. The runtime should not carry permanent exception lists for
defects the training recipe can simply stop producing. The measurement
below shows this is not optional taste: the magnitudes grow monotonically
with training, so **any fixed serving-side containment is a point-in-time
patch that a longer run walks past**.

## The incident

Value-truncated rollouts
([PR #106](https://github.com/eigen7/Scribblez/pull/106)) evaluate the
position evaluation model at rollout horizons, thousands of times per turn.
Under FP16 the `face-up-official` teacher (epoch 4414) deterministically
returned NaN in **every head** for certain legitimate inputs — post-bingo,
+150..+190-lead states that rollouts reach routinely. FP32 on the same rows
was sane, and the NaN reproduced for a single row evaluated alone: a
content-dependent overflow inside the engine, unfixable at decode time (by
the time any softmax/sigmoid we control runs, the values are already
destroyed).

## The measurements

Method: run the ONNX graph in FP32 with every intermediate tensor exposed
as an output (onnx `shape_inference` + appending `value_info` entries to
`graph.output`, onnxruntime CPU) over ~320 post-move rows selected for
extreme current-score leads plus a random slice; record peak absolute
values against FP16's max normal, **65504**.

**Where the overflow lives** (ep4414): the only violations are the trunk's
pooled-FC branch, in each pooled block (`blocks.{2,5,8}`) —
`pool_fc/Gemm` output peaks at ~72k at blocks.8, the broadcast `Add`
carrying it back into the trunk at ~73k; blocks.2/5 sit at 38–48k. The
values re-enter FP16 range only at the block's following `bn2`. Everything
else peaks below 29k.

**The peaks grow monotonically with training** (same probe batch, peak
|activation| per checkpoint):

| checkpoint               | pool branch | rest of net | wld logits |
|--------------------------|------------:|------------:|-----------:|
| face-up-official ep500   |       4,728 |       1,017 |        229 |
| face-up-official ep1000  |      10,044 |       2,372 |        437 |
| face-up-official ep2000  |      26,458 |       5,870 |      1,808 |
| face-up-official ep3000  |      45,251 |      12,773 |      5,723 |
| face-up-official ep4414  |  **73,169** |      28,837 |     23,350 |
| fixed5 ep47              |       1,315 |         827 |         15 |
| fixed2 ep949             |      28,504 |      25,644 |        740 |
| sd-mean-mse ep357        |       3,442 |       2,375 |         83 |

Roughly a doubling per ~1000 epochs, with FP16 range crossed between
ep3000 and ep4414 — and on this trajectory the **rest of the net**
(~29k and climbing) crosses too in a continued run, which is why pinning
the pool branch cannot be the durable answer. The growth is not unique to
one lineage (fixed2 is at 28k/25k by ep949); nothing in the objective
pushes back on it. The ±23k `wld` logits (a fully saturated softmax) are
part of the same unbounded growth; whether they interact with the
calibration drift of [pov_calibration_bias.md](pov_calibration_bias.md)
is **not established** — that document's corrected diagnosis attributes
its bias to an unpinned flat-direction walk and does not implicate
magnitudes. Treat these as two separately measured defects of the same
recipe.

## The current containment (to be retired)

PR #106 serves FP16 by pinning exactly the measured overflow region to
FP32 at engine-build time: the model family's spec declares overflow-prone
layer-name substrings (`PositionEvaluationSpec::kFp32LayerSubstrings =
{"/pool_fc/"}`, [model_specs.h](../engine/include/nn/model_specs.h)), and
the build ([neural_net.cpp](../engine/src/nn/neural_net.cpp),
`pin_fp32_region`) pins matched layers plus their downstream consumers
through the first re-normalizing layer — with FP32 output storage in
between, since a 72k value downcast mid-chain overflows just the same —
under `kOBEY_PRECISION_CONSTRAINTS`. Pinned FP16 plans get their own
engine-cache key; a substring matching no layer fails the build loudly;
and `SimRunner` hard-errors on any NaN leaf readout.

Validated at ep4414: zero NaNs on the workload that previously produced
86; observations within FP16 rounding of the FP32 reference (max 3.6e-4
win-prob / 0.09 pts); a 400-position sim run at 22s vs 41s with an FP32
leaf. But per the growth table, this is a patch fitted to one checkpoint's
overflow geography — ~120 lines of architecture-coupled exception
machinery whose only long-term justification would be serving checkpoints
we do not intend to keep.

## Why not bf16

bf16 (FP32's exponent range, 8-bit mantissa) would make serving
indifferent to magnitude growth with one builder flag. Rejected as the
steady state:

- **Permanent mantissa loss everywhere**: ~0.4% relative step vs FP16's
  ~0.05% — roughly ±0.8 pts per score-diff readout at ±200. It would
  coarsen not just sim leaves but near-tie candidate ranking and the
  teacher labels the student distills from (`.mset` generation serves the
  teacher at the family default precision).
- **It institutionalizes tolerance of unbounded growth**: the NaN
  tripwire is how the growth was found at all. A recipe whose activations
  double every thousand epochs has a defect worth an invariant, not a
  wider dynamic range to grow into.
- Decision rule for the future: if an architecture ever *legitimately*
  needs dynamic range beyond FP16 by design, revisit the serving format
  (bf16) then. Do not resurrect per-layer pins.

## The proposal

**1. Bound magnitudes in the training recipe.** Add a small activation
penalty on the pooled-FC pre-activations and/or a z-loss-style penalty on
the `wld` logits (`logsumexp`-squared, the standard form) so the objective
actively opposes growth instead of ignoring it. The bar is low: fixed5 and
sd-mean-mse sit 20–50x below FP16 range without any of this; the recipe
merely needs a restoring force against a walk it currently doesn't feel —
note the structural rhyme with the calibration drift's "add restoring
force" requirement, without assuming the two share a mechanism. Verify by
running the probe across the new run's checkpoints: peaks should plateau,
not double per thousand epochs.

**2. Gate it at export and promotion.** At ONNX export time (and at
teacher promotion under
[generational_teacher.md](generational_teacher.md)), run a probe batch —
including extreme-lead states, where magnitudes peak — through the
exported graph in FP32 with intermediates exposed (seconds on CPU), and
**fail the export** if any intermediate exceeds FP16 range with 4x
headroom (threshold 16384). Gate both families: the move-set student
shares the pooled-FC trunk structure and serves FP16 today with no check
at all. This can land **now**, before any recipe change — like the
calibration gates, observability first — and its per-epoch series on the
next run measures the growth spectrum for free.

**3. Retire the pins.** When the first gated teacher lands: delete
`kFp32LayerSubstrings` from both specs, `pin_fp32_region` and its
build/cache-key hooks in `neural_net.cpp`, and the `RuntimeSpec` plumbing
— one commit. The `SimRunner` NaN hard error stays permanently as the
belt-and-suspenders tripwire.

## Coordination

Self-contained: a recipe term, an export-time check, and a deletion. No
dependency on the calibration fix program in
[pov_calibration_bias.md](pov_calibration_bias.md) — its interventions
(loss shape, marginal-calibration terms, bounded averaging) and this
proposal's magnitude penalty are orthogonal recipe changes that can share
any future teacher run without gating each other. The gates from both
documents belong in the same promotion-gate set.

## Acceptance criteria

- The export gate passes on the new teacher (and student) with >= 4x
  FP16-range headroom across the run's checkpoints (plateau, not growth),
  and demonstrably fails on `face-up-official` ep4414 (the known-bad
  reference).
- A plain-FP16 build with the pin machinery deleted runs the PR #106
  validation workload (400 positions x 5 candidates x 200 truncated
  rollouts) with zero NaNs and observations within FP16 rounding of an
  FP32 reference.
- No match-strength regression attributable to the magnitude penalties
  (the match harness arbitrates, as always).
