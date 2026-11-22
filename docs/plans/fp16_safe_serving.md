# FP16 activation overflow: serve BF16

**Status: landed.** The engine serves the value models in BF16 by default.
The leaf network is the exception: it stays in FP16 for speed, with a
training-side restoring force ([The leaf network](#the-leaf-network)).

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

## The leaf network

DeepBot's and LeafBot's leaf network (`deep_bot_leaf_eval`) is evaluated once
per rollout, so its throughput is the search's leaf budget, and it is served
in FP16 (`--leaf-precision`). BF16 is not an option there: on the transformer
leaf, LeafBot's analysis of the known positions ran 15 to 18 times slower in
BF16 than in FP16, about as slow as FP32.

**The incident.** `deep_bot_leaf_eval/v7` (transformer trunk, width 64,
depth 4) returned NaN at generation 1410 for a bag-emptying bingo at one of
the known positions; FP32 on the same position was sane. Nothing caught it in
the engine: LeafBot's `predict_wld` path had no finiteness check, a NaN on the
first candidate won the pick, and the NaN reached the analysis JSON as `null`.

**The measurement.** The same probe as above, over 16k held-out rows. One
tensor grows without bound: the residual stream after the last layer (its MLP
output added in, `layers.3/down` into `Add_2`), whose only consumer is the
trunk's final LayerNorm. Everything else stays near 5k.

| generation | 100 | 400 | 700 | 1000 | 1200 | 1400 | 1700 | 1990 |
|------------|----:|----:|----:|-----:|-----:|-----:|-----:|-----:|
| peak       | 335 | 2.0k | 5.3k | 26k | 45k | **69k** | 97k | 152k |

This is the same mechanism as the incident above. A normalization layer
discards its input's scale, so no task loss opposes the growth of a tensor
that feeds one: the face-up teacher's `pool_fc` branch fed `bn2`, and v7's
residual stream feeds the final LayerNorm.

**The fix.**
- **A restoring force on exactly the scale the loss cannot see.** The leaf
  model adds `lambda_norm_input` (default 1e-6) times the mean square of every
  normalization layer's input to its loss (`scribblez/norm_inputs.py`). The
  task losses are indifferent to those scales, so the penalty holds them down
  without trading against the fit while they stay small. It covers both
  trunks: the transformer's residual stream and the conv tower's pre-BatchNorm
  activations. On v7's exports the term would have been about 7e-4 at
  generation 100 (under 0.1% of a total loss near 0.87) and about 1.1 at
  generation 1000 (peak 26k).

  Shrinking a scale that has already run away is not free. From v7's
  generation-1993 checkpoint, where the penalty began at about 15 times the
  whole loss, 1500 batches of AdamW (lr 1e-4) took the peak from 152k to 42k,
  and held-out WLD NLL rose from 0.4647 to 0.5038, against 0.4675 for the same
  batches without the penalty. A fresh run's penalty starts near 7e-4, so its
  cost is what that run's series shows.
- **The peak as a series.** Each generation's evaluation records
  `activation_peak`, the largest input or output of any module on the
  held-out rows in FP32 (`scribblez/activation_peak.py`); deep_bot records it
  as `leaf_activation_peak`. It should plateau far below 65504.
- **A hard error on a non-finite leaf.** Every read of the leaf's WLD in the
  GPU leaf service (`GpuRolloutValues`) refuses a non-finite logit, so an
  overflow fails the first game that meets it rather than leaking into picks
  and labels.
