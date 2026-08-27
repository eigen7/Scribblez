# FP16 activation overflow — resolved by serving BF16

**Status: resolved.** The position/move-set value model's activations grow
past FP16's range during training and NaN when served in FP16. The durable
fix is to **serve BF16**, whose exponent range is FP32's: the overflow cannot
occur, at a mantissa cost far below the model's own error. The model-side
containment program this document once proposed — activation-magnitude
penalties, an export/promotion gate, and per-layer FP32 pins — has been
**retired**; this record keeps the incident, the measurements, and why the
serving-format switch won out.

## The incident

Value-truncated rollouts
([PR #106](https://github.com/eigen7/Scribblez/pull/106)) evaluate the
position-evaluation model at rollout horizons, thousands of times per turn.
Under FP16 the `face-up-official` teacher (epoch 4414) deterministically
returned NaN in **every head** for certain legitimate inputs — post-bingo,
+150..+190-lead states that rollouts reach routinely. FP32 on the same rows
was sane, and the NaN reproduced for a single row evaluated alone: a
content-dependent overflow inside the engine, unfixable at decode time (by
the time any softmax/sigmoid we control runs, the values are already
destroyed).

## The measurements

Method: run the ONNX graph in FP32 with every intermediate tensor exposed as
an output (onnx `shape_inference` + appended `value_info`, onnxruntime CPU)
over ~320 post-move rows selected for extreme current-score leads plus a
random slice; record peak absolute values against FP16's max normal, 65504.

**The peaks grow monotonically with training** (same probe batch, peak
|activation| per checkpoint):

| checkpoint               | pool branch | rest of net | wld logits |
|--------------------------|------------:|------------:|-----------:|
| face-up-official ep500   |       4,728 |       1,017 |        229 |
| face-up-official ep1000  |      10,044 |       2,372 |        437 |
| face-up-official ep2000  |      26,458 |       5,870 |      1,808 |
| face-up-official ep3000  |      45,251 |      12,773 |      5,723 |
| face-up-official ep4414  |  **73,169** |      28,837 |     23,350 |

The overflow lives in the trunk's pooled-FC branch (`pool_fc/Gemm` output
~72k at blocks.8, the broadcast `Add` carrying it into the trunk ~73k),
values that re-enter FP16 range only at the block's following `bn2`. Nothing
in the objective pushes back on the growth, so "fits today" is not a
property any single-run patch could certify — a longer run walks past it.

## The resolution: serve BF16

BF16 carries FP32's 8-bit exponent (range ~3.4e38) with an 8-bit mantissa.
The overflow is a *range* problem, so BF16 removes it by construction: a 73k
activation is nowhere near BF16's range, and no future growth reaches it
either. The only cost is mantissa precision (~0.4% relative step vs FP16's
~0.05%), and that cost is immaterial to outcomes.

Measured on the real models (torch, FP32 reference vs FP16 and BF16 over the
1000-position large eval set, the 12 frozen positions, and the gate's
extreme-lead probe):

- **BF16 removes the overflow.** On the 73k-peak ep4414 model, full-cast BF16
  stays numerically clean — score-diff rel_p95 ~1–4%, and win-MAE /
  score-diff-MAE against Monte-Carlo truth **identical to FP32** (0.0526 /
  13.55 vs 0.0526 / 13.54). Full-cast **FP16** on the same model is
  catastrophically wrong: ~31-point score-diff error, rel_p95 ~253%, win-MAE
  degraded 0.014 → 0.061.
- **BF16's precision cost does not move any quality metric.** Across every
  dataset, BF16 adds ~0.1–0.4 pt mean / ~1–2 pt worst-case score-diff noise
  and ~5e-4–1.6e-3 win-probability noise — 20–50× below the model's own error
  against ground truth. The aggregate quality metrics are unchanged to three
  significant figures.

This reverses the earlier decision to reject BF16 on precision grounds: the
~0.4% relative step is real but demonstrably lost in the model's own
Monte-Carlo error, and it buys the elimination of an entire class of
serving-time failure with no per-run tuning, no growth policing, and no
architecture-coupled pin machinery. The `nn_inference_parity` /
`mset_inference_parity` engine tests run BF16 alongside FP16 against the
PyTorch FP32 reference to keep the served path honest.

## What changed

- **Serving.** `Precision::kBF16` added to the engine; the value-truncation
  leaf service (`load_leaf_position_service`) and the shared agent options
  (`NeuralServiceOptions`, default `BF16`) serve BF16.
- **Removed.** The FP32-pinning machinery (`kFp32LayerSubstrings`,
  `pin_fp32_region`, the `RuntimeSpec` plumbing and pinned cache-key branch);
  the FP16 export gate (`fp16_gate.py`, the probe builders, the per-export
  `fp16_probe_peak` metric); and the activation-magnitude recipe terms
  (`PoolFcPenalty`/`lambda_pool_act` and `wld_z_loss`/`lambda_wld_z`) in all
  three trainer families.
- **Kept.** `SimRunner`'s non-finite-leaf hard error stays permanently as the
  belt-and-suspenders tripwire: under BF16 it should never fire, so a trip
  now means an off-distribution input or a genuinely broken model.
