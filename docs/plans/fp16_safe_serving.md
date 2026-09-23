# FP16 activation overflow: serve BF16

**Status: landed.** The engine serves the value models in BF16 by default.

**Problem.** The position and move-set value models' activations grow past
FP16's range as training runs on, and a model served in FP16 then returns NaN.

**Decision.** Serve BF16. Its exponent range is FP32's, so the overflow cannot
occur, and its mantissa cost is far below the model's own error. A
model-side containment program (activation-magnitude penalties, an
export/promotion gate, per-layer FP32 pins) was built first and then removed
once the BF16 measurements below were in. This record keeps the incident, the
measurements, and why changing the serving format beat containing the model.

## The incident

Value-truncated rollouts
([PR #106](https://github.com/eigen7/Scribblez/pull/106)) evaluate the
position evaluation model at rollout horizons, thousands of times per turn.
Under FP16 the `face-up-official` teacher (epoch 4414) returned NaN in every
head, deterministically, for certain legitimate inputs: post-bingo states
with a +150 to +190 lead, which rollouts reach routinely. FP32 on the same
rows was sane, and a single row evaluated alone reproduced the NaN. The
overflow happens inside the TensorRT engine and is content-dependent; it
cannot be repaired at decode time, because the values are already destroyed
before any softmax or sigmoid we control runs.

## The measurements

Method: run the ONNX graph in FP32 with every intermediate tensor exposed as
an output (onnx shape inference plus appended `value_info`, onnxruntime on
CPU) over about 320 post-move rows, chosen for extreme current-score leads
plus a random slice. Record each tensor's peak absolute value against FP16's
largest normal, 65504.

**The peaks grow monotonically with training** (same probe batch, peak
|activation| per checkpoint):

| checkpoint               | pool branch | rest of net | wld logits |
|--------------------------|------------:|------------:|-----------:|
| face-up-official ep500   |       4,728 |       1,017 |        229 |
| face-up-official ep1000  |      10,044 |       2,372 |        437 |
| face-up-official ep2000  |      26,458 |       5,870 |      1,808 |
| face-up-official ep3000  |      45,251 |      12,773 |      5,723 |
| face-up-official ep4414  |  **73,169** |      28,837 |     23,350 |

The overflow sits in the trunk's pooled-FC branch: the `pool_fc` Gemm output
reaches about 72k at block 8, and the broadcast Add that carries it into the
trunk about 73k. The values come back into FP16 range only at the block's
following `bn2`. Nothing in the training objective pushes back on this
growth, so "fits in FP16 today" cannot be certified for a run: a longer run
walks past it.

## Why BF16

BF16 has FP32's 8-bit exponent (range about 3.4e38) and an 8-bit mantissa.
The overflow is a range problem, so BF16 removes it by construction: a 73k
activation is nowhere near its limit, and neither is any plausible future
growth. The cost is precision, a relative step of about 0.4% against FP16's
0.05%, and that cost does not reach outcomes.

Measured in torch on the real models, FP32 reference against FP16 and BF16,
over the 1000-position large eval set, the 12 frozen positions, and an
extreme-lead probe set:

- **BF16 removes the overflow.** On the 73k-peak ep4414 model, full-cast BF16
  stays numerically clean: score-diff rel_p95 about 1 to 4%, and win-MAE /
  score-diff-MAE against Monte-Carlo truth identical to FP32 (0.0526 / 13.55
  against 0.0526 / 13.54). Full-cast FP16 on the same model is badly wrong:
  about 31 points of score-diff error, rel_p95 about 253%, win-MAE degraded
  from 0.014 to 0.061.
- **BF16's precision cost moves no quality metric.** Across every dataset,
  BF16 adds about 0.1 to 0.4 points mean and 1 to 2 points worst-case
  score-diff noise, and 5e-4 to 1.6e-3 of win-probability noise: 20 to 50
  times below the model's own error against ground truth. The aggregate
  quality metrics agree with FP32 to three significant figures.

BF16 had been set aside earlier on precision grounds. The measurement shows
that its 0.4% step disappears inside the model's Monte-Carlo error, and in
exchange it eliminates a whole class of serving-time failure with no per-run
tuning, no policing of activation growth, and no pin machinery coupled to
the architecture. The engine tests `nn_inference_parity` and
`mset_inference_parity` run BF16 alongside FP16 against the PyTorch FP32
reference, which keeps the served path honest.

## What landed

- **Serving.** `Precision::kBF16` in the engine (`nn/trt_util.h`), and BF16
  as the default of `NeuralNetParamsBase::precision` (`nn/neural_net.h`), of
  the agents' shared `NeuralServiceOptions`, and of the value-truncation leaf
  service (`load_leaf_position_service`). Tools that build their params
  directly, such as the move-set-eval target generator and the evidence
  trajectory generator, therefore serve BF16 without opting in.
  The move-proposal graphs (`agent/move_proposal_nets.h`) are the exception:
  they default to FP32 for their parity contract.
- **Guards against non-finite outputs.** `SimRunner` raises a hard error on
  a non-finite leaf readout; under BF16 it should never fire, so a trip means
  an off-distribution input or a broken model. The move-set-eval target
  generator refuses to write a non-finite teacher readout, and the move-set
  trainer refuses a corpus it would have to drop whole. (Both exist because
  the generator once served a transformer teacher in FP16 and wrote an
  entire corpus of NaN targets.)
- **Removed.** The containment program in full: the FP32 layer pins
  (`kFp32LayerSubstrings`, `pin_fp32_region`, their `RuntimeSpec` plumbing
  and the pinned cache-key branch); the FP16 export gate (`fp16_gate.py`, its
  probe builders, the per-export `fp16_probe_peak` metric); and the
  activation-magnitude loss terms (`PoolFcPenalty` / `lambda_pool_act` and
  `wld_z_loss` / `lambda_wld_z`) in all three trainer families.
