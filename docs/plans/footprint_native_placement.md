# Footprint-native placement

**Status: landed** (PR A, BC1, BC2). Placement is footprint-categorical end to
end:

- The position evaluation **teacher** predicts each placement head as a
  categorical distribution over 2927 footprint classes
  (`engine/include/training/footprint.h`).
- The move set evaluation **student** distills a dense, board-masked footprint
  target from `.mset` v3 (BC1).
- The **evidence path** is footprint-native (BC2). The sim observation is a
  dense per-head footprint histogram (`.sobs` v5), and the evidence fusion
  consumes footprint slot-channel planes (`NUM_EVIDENCE_PLANES` 9 → 117).

Collapsing footprints to per-cell values survives only as a display
reduction, in two places: `footprint_cell_marginal` for the Trajectories pane,
and `collapse_footprint_planes` for the per-move analysis view.

PR D, the retraining, is operational rather than code. Footprint teachers have
since been trained (for example `transformer-clipped`) and a move-set tag
distills from one. No `.sobs` v5 evidence corpus had been generated as of
this writing.

**Goal.** Remove the per-cell collapse of the placement distribution
everywhere except visualization, so no consumer trains on or conditions on a
lossy summary.

**Decision.** Treat footprints as spatial: reshape the anchored classes onto
the board as 13 channels per square, so each consumer widens its channel axis
instead of changing its architecture. Store the `.mset` target and the
`.sobs` histogram dense. Keep the evidence path's spatial conv, widened.

This plan went through a four-panelist plan review, including a cross-vendor
seat. The design below is the post-review version. Where BC2's execution
departed from it, with approval, the departure is recorded inline as a
**BC2 deviation**; the review's resolved dissent is summarized at the end.

## The key idea: footprints are spatial

An anchored footprint class factors as `(cell, slot)`, with
`cell = r*15 + c` and `slot ∈ [0, 13)`: one orientation-free slot for k=1,
then k=2..7 horizontal, then k=2..7 vertical. The 2925 anchored classes
therefore **reshape losslessly to a `(15, 15, 13)` tensor**, the board grid
with 13 channels per square. The two remaining classes (`pass`, and the win
heads' "not-win" class) are non-spatial scalars.

That is why the migration widens rather than rearchitects. The sim
observation, the teacher target, the student head and the evidence conv all
keep their spatial shape and gain a 13-slot channel axis. The shared Python
primitive is `py/scribblez/footprint_spatial.py` (`to_spatial` /
`from_spatial`, and the sparse codec); the C++ side of the evidence layout is
`agent/evidence_staging.h`.

**Frame invariant (load-bearing).** Every placement path is pinned to the
game's natural frame, with no symmetry transpose. A diagonal transpose swaps
rows with columns *and* the horizontal slot channels with the vertical ones,
so the slot axis permutes too, not only H and W. Any future consumer that
transposes must permute the 13 channels as well.

## Storage, per format

A dense 2927-wide distribution is about 13× the per-cell marginal it replaces
(`.mset` goes from 900 B to about 11.7 KB per candidate). That motivated a
sparse top-k `(class: u16, value)` encoding (`footprint_spatial.top_k_sparse`).
But the two formats hold different distributions, and the measurement
contradicts the "peaked" assumption for one of them.

- **`.mset` teacher target: dense.** The offline probe
  `py/scripts/position_eval/footprint_topk_fidelity.py` measures the
  student's actual distillation target: the teacher's footprint softmax
  under the board-legality mask, computed through
  `ffi.masked_position_eval_placement` (the same mask and masked softmax the
  `.mset` writer applies). On a `footprints-official` checkpoint the masked
  distribution is **broad**. Top-128 keeps a p10 worst case of only about
  0.81 to 0.94 of the mass across heads, and `self_next_placement` is the
  most diffuse (median 0.90, p10 0.81 at k=128). No k ≤ 128 clears 99%, and
  capturing the tail would need k in the hundreds, which defeats the purpose
  of sparsity. So the `.mset` target is stored dense: per head, absmax
  quantized like the old per-cell planes but 2927 wide. It is also the
  simpler design (no codec, no k, no fidelity loss). The 13× size is the
  accepted cost; the corpus is regenerated anyway.
- **`.sobs` sim-observation histogram: dense (planned sparse).** This is a
  histogram of moves the rollouts actually played. The rollout policy is
  near-greedy per drawn rack, so about 300 rollouts touch few distinct
  footprints however broad the teacher's predicted distribution is. The plan
  was sparse top-k at a fixed padded width.

  *BC2 deviation: stored dense.* Keeping `SimObservation` a verbatim
  fixed-stride POD avoids a format rewrite: v4 → v5 is an ordinary version
  bump over the same layout machinery. The 13× is accepted on disk, and a
  sparse re-encode remains a later, purely mechanical change. The 13× must
  not reach trainer-resident RAM, though: a corpus at the scale of earlier
  runs, held dense, is about 58 GiB on a 62 GB host. So `TrajectoryDataset`
  repacks the four histograms sparse when it loads them and densifies per
  batch (`_PackedObs`, in `py/scribblez/evidence/dataset.py`).

Where a `k` is used it is a format constant, not a runtime tunable. The
project carries no backwards-compatibility burden, so each format commits to
one encoding; there is no dual dense/sparse path.

The planned sparse `.sobs` would have required **accumulating dense and
sparsifying on write**: the in-memory accumulator (`accumulate_rollout` in
`sim/sim_runner.h`) cannot know the top k until every rollout is folded in,
so the in-memory `SimObservation` stays a dense histogram and only the
serialized record is padded top-k. That makes the record no longer a
verbatim POD, which is a format rewrite (new `sizeof` and `static_assert`s)
rather than a version bump. Storing dense made the rewrite unnecessary.

## Win-head normalization

The two win heads reserve mass in `kExtraClass` ("not-win"); the plays heads
do not.

**Planned:** store predicted win channels **conditional on winning**: drop
`kExtraClass`, renormalize the placement mass to 1, and carry `P(win)` as a
scalar. Normalize the observed win histogram the same way (divide by its win
count to get `P(footprint | win)`, with observed `P(win)` as its scalar). The
matched normalization is what makes the per-head residual (observed minus
predicted) well-defined. Catch-all scalars would carry `P(win)` and the pass
mass per head.

*BC2 deviation: no renormalization and no catch-all scalars.* The evidence
planes carry the raw quantities: observed counts divided by rollouts, and the
unmasked softmax with the two catch-all classes dropped. Observed and
predicted are already in matched units (both un-renormalized frequencies over
the same class space); the conv can learn any residual scaling; and the WLD
scalars on each evidence token already supply `P(win)` on both sides. This
gives up the clean per-head residual for a simpler staging path with no
division.

## Predicted placement at serving is unmasked

The evidence path feeds the **unmasked** footprint softmax
(`footprint_slot_planes` in `py/scribblez/move_set_eval/model.py`: softmax
over all classes, catch-alls dropped, reshaped to slot channels). That is
pure arithmetic: no Board or Dictionary in evidence staging and no
per-candidate mask on the serving hot path. The board-legality mask stays a
training-target concern only (the teacher target and the student's masked
softmax-CE).

## The evidence path: one design fork

The per-cell "hotness" the old evidence conv consumed can be recovered from a
footprint head as a one-line slot sum,
`softmax(logits)[..., :2925].reshape(k, 4, 225, 13).sum(-1)`. That is what
let the student migrate (BC1) before the evidence path changed. For the
evidence path itself (BC2), the plan left a fork to benchmark:

- **Option 1: widened spatial conv (baseline).** Stack observed, predicted
  and candidate as `(15,15,13)`-derived channels (`NUM_EVIDENCE_PLANES`
  9 → 117) and keep the conv. Preserves the neighborhood bias; costs a dense
  117×15×15 staging tensor per evidence token (zeroed, then projected to
  about 32 features).
- **Option 2: compact sparse residual encoder.** Form `observed − predicted`
  per head, keep the candidate as a class index, and feed the sparse
  `(class, value)` entries through a learned anchor/slot-factorized embedding
  and pooling into the existing evidence attention. Avoids the dense tensor
  and makes the residual explicit; loses the conv's neighborhood bias and
  adds gather/scatter.

*BC2 deviation: Option 1, without the benchmark.* The widened conv is the
conservative choice: it keeps the working architecture and its neighborhood
bias, and the staging cost is a per-token 117×225 fill on a path that is not
hot. The compact encoder remains available as an experiment if the dense
tensor ever shows up in a profile.

## Visualization: the only surviving collapse, on two data paths

1. **Per-move analysis view** (`training/position_eval_analysis.cpp` →
   `scribblez_ffi.cpp` → `ffi.py` → `dashboard/api.py`). Runs the collapse on
   the teacher's live inference logits for a typed-in GCG. Its input did not
   change, so the corpus and format work could not affect it.
2. **Trajectories / Positions tab** (`sim_evidence/sobs.py`,
   `evidence/trajectory_view.py`, `dashboard/trajectories_api.py`). Reads
   `.sobs` placement directly as a numpy dtype, bypassing the collapse. Under
   `.sobs` v5 the pane's observed truth plane is the histogram's slot-sum
   anchor marginal, drawn beside the prediction's `footprint_cell_marginal`,
   so both per-cell views are the same reduction.

## PR slicing

Sequenced after the move-proposal subset-assembly dataset (#138), which
shares the evidence training-loop files.

- **PR A: shared primitive, fidelity probe, this note.** Landed. Inert:
  `footprint_spatial.py` and its tests, the offline `k` probe, this document.
- **PR BC1: student first, evidence path unchanged.** Landed. `.mset` v3
  with the teacher footprint target; the generator's collapse removed; the
  student and proposal plane heads become footprint heads trained by masked
  softmax-CE / KL. A slot-sum marginal fed the existing per-cell evidence
  path, so `.sobs`, `sim_runner.h`, evidence staging and the parity tests
  were untouched. Gated on student recall@1 and regret against the per-cell
  BCE baseline, which proves the core hypothesis before any format or
  evidence risk.
- **PR BC2: footprint-native evidence.** Landed, with the deviations above.
  The `.sobs` v5 histogram and evidence staging changed together; the
  evidence fusion migrated; both visualization data paths and
  `EvidencePositionEvalModel` / the kill test widened; proposal export and
  parity tests updated. Only worth doing once BC1's gates justified it.
- **PR D: corpus regeneration, retraining, gates.** Operational (see
  Status). BC1 retrains the student on the existing `.sobs`; a `.sobs` v5
  corpus is needed only for the evidence path.

## Decisions

1. **Storage per format**, settled by the masked fidelity probe: the `.mset`
   teacher target is dense (the masked distribution is broad; no k ≤ 128
   clears 99%). The `.sobs` histogram was planned sparse and landed dense
   (BC2 deviation: the verbatim POD wins). No dual dense/sparse path.
2. **Masked softmax-CE / KL** distillation loss, replacing per-cell BCE.
3. **Catch-all classes as scalars, win heads conditional on winning:**
   superseded by the BC2 deviation. The catch-alls are dropped outright and
   the planes carry raw, un-renormalized histograms and distributions.
4. **Fusion:** the widened spatial conv, committed without the
   compact-encoder benchmark (BC2 deviation).
5. **Predicted placement is unmasked at serving**: no Board or Dictionary in
   evidence staging.

## Plan-review dissent (resolved)

- **Split the migration and measure before fixing the format** (three
  panelists, including cross-vendor): adopted as the BC1/BC2 split and the
  offline `k` probe.
- **Sparse storage breaks the fixed-stride POD, and the accumulator cannot
  sparsify mid-fold:** adopted (dense accumulator, sparsify on write, format
  rewrite), then made moot when `.sobs` stayed dense.
- **Masking would land on the serving path:** adopted; unmasked at serving.
- **Win-head `kExtraClass` normalization mismatch:** adopted as
  conditional-on-win, then superseded by the raw-quantities deviation.
- **The `.sobs` v5 blast radius missed the Trajectories tab readers**
  (blocking): adopted; added to BC2.
- **A compact residual encoder over the dense 117-plane conv:** partially
  adopted. The conv stayed as the baseline with the encoder to be
  benchmarked in BC2; BC2 then committed the conv without the benchmark.
- **`EvidencePositionEvalModel` and the kill test at the same seam:** kept
  in scope and widened in BC2. They are the diagnostic that measures the
  evidence signal, worth keeping alive across the change.
