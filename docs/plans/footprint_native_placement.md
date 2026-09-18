# Footprint-native placement

The position-eval **teacher** predicts placement as a categorical distribution
over 2927 **footprint** classes (`engine/include/training/footprint.h`). The
move-set-eval **student**'s distillation target is footprint-native too (BC1,
below), and BC2 migrated the **evidence path**: the sim observation is a dense
per-head footprint histogram (`.sobs` v5), the evidence fusion consumes
footprint slot-channel planes (`NUM_EVIDENCE_PLANES` 9 → 117), and the
transitional per-cell bridge (`footprint_cell_marginal` feeding the fusion) is
gone -- the marginal survives only as a display reduction (the Trajectories
pane), alongside `collapse_footprint_planes` for the per-move analysis viz.

This document is the plan that drove the migration. It was reviewed by a
four-panelist plan-review (including a cross-vendor seat); the design below is
the post-review version, the dissent it resolved is summarized at the end, and
the **BC2 execution deviations** -- simplifications approved after review --
are recorded inline in *[BC2: ...]* notes.

## The key idea: footprints are spatial

An anchored footprint class factors as `(cell, slot)` with `cell = r*15 + c` and
`slot ∈ [0, 13)` (1 orientation-free k=1 slot, then k=2..7 horizontal, then k=2..7
vertical). So the 2925 anchored classes **reshape losslessly to a `(15, 15, 13)`
spatial tensor** — the board grid with 13 channels per square — and the 2
catch-all classes (`pass`, the win heads' not-win/dummy) are non-spatial scalars.

This is why the migration is a *widening*, not a rearchitecture: the sim
observation, the teacher target, the student head, and the evidence conv all keep
their spatial shape and gain a 13-slot channel axis. The shared primitive is
`py/scribblez/footprint_spatial.py` (`to_spatial`/`from_spatial`, the sparse
codec); the C++ mirror lands with its first writer.

**Frame invariant (load-bearing):** every placement path is pinned to the game's
natural frame (no symmetry transpose). A diagonal transpose swaps rows↔cols
**and** the horizontal↔vertical slot channels, so the *slot axis* — not only H/W
— permutes under a transpose. Any future transposed consumer must permute the 13
channels too.

## Storage: sparse vs. dense, per format

A dense 2927-wide plane is ~13× the per-cell marginal it replaces (`.mset`
900 B → ~11.7 KB/candidate), which motivated a sparse top-k `(class:u16, value)`
encoding (`footprint_spatial.top_k_sparse`). But the two formats hold different
distributions and the offline probe already contradicts the "peaked" assumption
for one of them:

- **`.mset` teacher target — dense.** The offline probe
  `py/scripts/position_eval/footprint_topk_fidelity.py` measures the fidelity of
  the student's actual distillation target: the engine's board-legality-**masked**
  footprint softmax (via `ffi.masked_position_eval_placement`, which runs the same
  mask + masked-softmax the `.mset` writer applies). On a footprints-official
  checkpoint the masked distribution is **broad** — top-128 keeps a p10 worst-case
  of only ~0.81–0.94 across the heads, and `self_next_placement` is the most
  diffuse (median 0.90, p10 0.81 at k=128). No k ≤ 128 clears 99%; capturing the
  tail would need k in the many hundreds, defeating the point of sparsity. So the
  `.mset` teacher target is stored **dense** (per head, absmax-quantized like the
  current per-cell planes but 2927-wide) — which is also the simpler design (no
  codec, no k, no fidelity loss), at ~13× the per-cell planes' bytes. That size is
  the accepted cost; the corpus regenerates anyway.
- **`.sobs` sim-obs histogram — sparse.** This is a histogram of *actual* rollout
  moves; the rollout policy is near-greedy per drawn rack, so ~300 rollouts touch
  few distinct footprints regardless of how broad the teacher's *predicted*
  distribution is. Sparse top-k with a fixed padded width fits; confirm the width
  against a real histogram when `.sobs` v5 exists (BC2).
  *[BC2 deviation: stored **dense** after all. Keeping `SimObservation` a
  verbatim fixed-stride POD avoids the format rewrite entirely (a version bump
  over the same layout machinery); the ~13× size is accepted on disk, and a
  sparse re-encode stays open as a later, purely mechanical change. The 13×
  must NOT reach trainer-resident RAM, though — a precedent-scale corpus held
  dense is ~58 GiB on a 62 GB host — so `TrajectoryDataset` repacks the four
  histograms sparse at absorb and densifies per batch (`_PackedObs`).]*

`k` (where used) is a **format constant**, not a runtime tunable. The project
carries no backwards-compatibility burden, so each format commits to one encoding
directly; there is no dual dense/sparse path.

**Accumulate dense, sparsify on write.** `SimRunner::accumulate` can't know the
top-k until every rollout is folded in, so the in-memory `SimObservation` stays a
dense per-head histogram; only the serialized record is padded top-k. The `.sobs`
and `.mset` writers/readers therefore stop treating the observation/planes as a
verbatim fixed-stride POD — a format rewrite (new `sizeof` + `static_assert`s),
not a version-number bump over the same layout. Same for `.mset`'s `TargetWriter`.
*[BC2: with `.sobs` dense, none of this rewrite was needed — the record stays a
verbatim POD and v4 → v5 is an ordinary version bump.]*

## Win-head normalization

The two win heads reserve mass in `kExtraClass` ("not-win"); the plays heads do
not. Predicted win channels are stored **conditional-on-win**: drop `kExtraClass`,
renormalize the placement mass to sum to 1, and carry `P(win)` as a scalar. The
observed win histogram (`accumulate` adds `p_win`/`p_loss` into the reply's
footprint bucket) is normalized the **same** way — divided by its win count to
`P(footprint | win)` — with observed `P(win)` as its scalar. Matched
normalization is what makes the evidence residual (observed vs. predicted)
well-defined. Catch-all scalars: `P(win)` and pass mass per head; the plays heads'
extra slot is inert.

*[BC2 deviation: no conditional-on-win renormalization, and no catch-all
scalars. The evidence planes carry the RAW quantities — observed counts / n and
the unmasked softmax with the two catch-all classes simply dropped — because
observed and predicted are already in matched units (both un-renormalized
frequencies over the same class space), the conv learns any residual scaling,
and the WLD scalars the token carries already supply `P(win)` on both sides.
This trades the clean per-head residual for a simpler, division-free staging
path.]*

## Predicted placement at serving is unmasked

The evidence path feeds the **unmasked** footprint softmax
(`footprint_slot_planes`: softmax over all classes, catch-all dropped, reshaped
to slot channels) — pure arithmetic, no Board or Dictionary in
`evidence_staging`, and no per-candidate mask on the serving hot path. The
board-legality mask remains a **training-target** concern only (the teacher
target and the student's masked softmax-CE), exactly where it lives today.

## The evidence path: one design fork

The per-cell "hotness" the evidence conv consumes is recoverable from a footprint
head as `softmax(logits)[..., :2925].reshape(k, 4, 225, 13).sum(-1)` — a one-line
slot-sum marginal. That is what lets the student migration land *before* the
evidence path changes (see PR BC1). When the evidence path is itself migrated
(BC2), there is a fork to **benchmark**, not decide up front:

- **Option 1 — widened spatial conv (baseline).** Stack observed/predicted/candidate
  as `(15,15,13)`-derived channels (`NUM_EVIDENCE_PLANES` grows 9 → ~117), keep the
  conv. Preserves the neighborhood bias; costs a dense ~117×15×15 staging tensor per
  evidence slot (zeroed, then projected to ~32 features).
- **Option 2 — compact sparse residual encoder.** Form `observed − predicted` per
  head, keep the candidate as a class index, feed the sparse `(class, value)`
  entries through a learned anchor/slot-factorized embedding + pooling into the
  existing evidence attention. Avoids the dense tensor and is explicit about the
  residual; loses the conv's neighborhood bias, adds gather/scatter.

Benchmark Option 2 against Option 1 and the collapsed baseline in BC2.

*[BC2 deviation: Option 1 committed without the benchmark. The widened conv is
the conservative choice (it preserves the working architecture and its
neighborhood bias) and the staging cost is a per-token 117×225 fill on a path
that is not hot; the compact encoder remains available as a later experiment if
the dense tensor ever shows up in a profile.]*

## Visualization is the only surviving collapse — but two data planes

1. **Per-move analysis viz** (`position_eval_analysis.cpp → scribblez_ffi.cpp →
   ffi.py → dashboard/api.py`) runs the collapse on the teacher's **live inference
   logits** for a typed-in GCG. Its input is unchanged, so it is provably
   unaffected by the corpus/format work — not a regen gate.
2. **Trajectories/Positions tab** (`sim_evidence/sobs.py`,
   `evidence/trajectory_view.py`, `dashboard/trajectories_api.py`) reads `.sobs`
   placement **directly** (numpy dtype), bypassing the collapse. `.sobs` v5
   changed it under BC2: the pane's observed truth plane is now the histogram's
   slot-sum anchor marginal, drawn beside the prediction's
   `footprint_cell_marginal` — the two per-cell views are now the same
   reduction on both sides.

## PR slicing

Sequenced after the move-proposal subset-assembly dataset (#138), which shares the
evidence train-loop files.

- **PR A — shared primitive + fidelity probe + this note** (here). Inert:
  `footprint_spatial.py`, its tests, the offline `k` probe, the design doc. No
  behavior change.
- **PR BC1 — student-first, evidence path unchanged.** `.mset` v3 teacher footprint
  target; remove the generator collapse; student/proposal plane head → footprint +
  masked softmax-CE / KL loss; a one-line slot-sum marginal feeds the *existing*
  per-cell evidence path, so `.sobs`, `sim_runner.h`, `evidence_staging`, and the
  parity tests are untouched. Gate on student recall@1 / regret vs. the per-cell
  BCE baseline — proves the core hypothesis before any format/evidence risk.
- **PR BC2 — footprint-native evidence.** `.sobs` v5 sim-obs histogram (dense
  accumulator → sparsify on write; the struct change and `evidence_staging` land
  atomically); evidence fusion migrated (Option 1 vs. Option 2 benchmarked); the
  two viz data planes and `EvidencePositionEvalModel`/`kill_test` widened; proposal
  export and parity tests. Only invest here once BC1's gates justify it.
- **PR D — corpus regen + retrain + gates.** The teacher is already footprint; the
  `.sobs` v5 corpus is needed only once BC2 lands (BC1 retrains the student on the
  existing `.sobs`).

## Decisions

1. **Storage is per-format** (settled by the masked fidelity probe): `.mset` teacher
   target **dense** (the masked distribution is broad — no k ≤ 128 clears 99%);
   `.sobs` sim-obs histogram — planned sparse top-k, **landed dense** (BC2
   deviation above: the verbatim POD wins). No dual dense/sparse path either way.
2. **Masked softmax-CE / KL** distillation loss (replaces per-cell BCE).
3. **Catch-all → scalars**, win heads **conditional-on-win** — *superseded by the
   BC2 deviation above: catch-all dropped outright, raw un-renormalized
   histograms/distributions.*
4. **Fusion:** the widened spatial conv, committed without the compact-encoder
   benchmark (BC2 deviation above).
5. **Predicted placement at serving is unmasked** (no Board/Dictionary in staging).

## Plan-review dissent (resolved)

- **Split the migration + measure before baking the format** (three panelists,
  incl. cross-vendor): adopted — BC1/BC2 split and the offline `k` probe.
- **Sparse breaks the fixed-stride POD; accumulate can't sparsify mid-fold**:
  adopted — dense accumulator, sparsify on write, format rewrite.
- **Masking would land on the serving path**: adopted — unmasked at serving.
- **Win-head `kExtraClass` normalization mismatch**: adopted — conditional-on-win.
- **`.sobs` v5 blast radius missed the Trajectories tab readers** (blocking):
  adopted — added to BC2.
- **Compact residual encoder over the dense 117-plane conv**: partially adopted —
  kept the conv as the baseline, benchmark the encoder in BC2.
- **`EvidencePositionEvalModel`/`kill_test` at the same seam**: kept in scope —
  widened in BC2 (it is the diagnostic that measures the evidence signal, worth
  keeping alive across the change).
