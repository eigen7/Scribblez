#!/usr/bin/env python3
"""The roadmap-item-3 TRT gate: prove the two move-proposal ONNX graphs parse,
build, and REFIT correctly under TensorRT before the engine runtime is designed
around them (the plan's front-loaded #1 risk).

The cache graph is close to the mset graph (already covered by trt_refit_probe),
so the point here is the STEP graph: its fusion stage carries the pieces that
have never been exported before -- the split self-attention MHA, the additive
padding-bias softmaxes, the TransformerEncoderLayer LayerNorms + FFN, and the 4D
`einsum` mixing the spatial features. A silently wrong refit of any of them, or
an op TensorRT cannot build, would surface here rather than in the C++ runtime.

On the GPU, for EACH graph:
  1. Export two random-init production-shape proposal models A and B.
  2. Parse A and build a refittable FP32 engine with the dynamic-M profile the
     C++ runtime will use (the step graph's evidence inputs are fixed-width, so
     only move_enc / the move inputs ride M).
  3. Refit that engine with B's weights; assert no weights go missing.
  4. Run the refitted engine at several Ms and compare against ONNXRuntime on B
     -- a wrong mapping makes the outputs A/B chimeras and the comparison fails.

Exit code 0 = gate passed.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scribblez.evidence_fusion import NUM_EVIDENCE_PLANES, NUM_EVIDENCE_SCALARS
from scribblez.ffi import get_input_shapes
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_dims
from scribblez.move_set_eval.proposal_export import (
    DEFAULT_MAX_EVIDENCE,
    export_proposal_cache,
    export_proposal_step,
    proposal_export_id,
)
from scribblez.move_set_eval.targets import PLANE_NAMES

MAX_MOVES = 4096
OPT_MOVES = 512
PROBE_MS = (1, 37, 512)
# FP32 TRT vs FP32 ORT is kernel-order noise, not semantics; a wrong refit makes
# the outputs A/B chimeras that differ by O(1) relative, far above either bound.
# The scalar heads are small and bounded, so an absolute bound fits them; the
# raw plane logits are unbounded (O(tens)) and sum over C*225 in a different
# order than ORT, so they are judged by RELATIVE error against ORT's magnitude.
TOLERANCE = 2e-3
PLANES_RTOL = 5e-2
BOARD = 15


def _shapes():
    dims = {s.name: s.dims for s in get_input_shapes()}
    return dims["input_spatial"][0], dims["input_scalar"][0]


def _random_model(seed: int, spatial_planes: int, scalar_size: int) -> MoveSetEvalModel:
    torch.manual_seed(seed)
    model = MoveSetEvalModel(spatial_planes, scalar_size)  # production dims
    with torch.no_grad():
        for module in (model.evidence_fusion, model.proves_best):
            for p in module.parameters():
                p.add_(0.1 * torch.randn_like(p))
    model.eval()
    return model


def _cache_inputs(m: int, spatial_planes: int, scalar_size: int, seed: int) -> dict:
    t, s, letter_vocab, cells = move_encoding_dims()
    rng = np.random.default_rng(seed)
    return {
        "input_spatial": rng.standard_normal((1, spatial_planes, BOARD, BOARD), dtype=np.float32),
        "input_scalar": rng.standard_normal((1, scalar_size), dtype=np.float32),
        "move_letters": rng.integers(0, letter_vocab, (m, t), dtype=np.int32),
        "move_blanks": rng.integers(0, 2, (m, t)).astype(np.uint8),
        "move_squares": rng.integers(0, cells, (m, t), dtype=np.int32),
        "move_tile_mask": rng.integers(0, 2, (m, t)).astype(np.uint8),
        "move_scalars": rng.standard_normal((m, s)).astype(np.float32),
    }


def _step_inputs(m: int, channels: int, seed: int) -> dict:
    e = DEFAULT_MAX_EVIDENCE
    rng = np.random.default_rng(seed)
    k = min(m, e // 2)  # some real evidence, some padding
    mask = np.zeros((1, e), np.uint8)
    mask[0, :k] = 1
    return {
        "board": rng.standard_normal((1, BOARD * BOARD, channels), dtype=np.float32),
        "g": rng.standard_normal((1, 3 * channels), dtype=np.float32),
        "move_enc": rng.standard_normal((m, channels), dtype=np.float32),
        "ev_move_enc": rng.standard_normal((1, e, channels), dtype=np.float32),
        "ev_obs_planes": np.abs(
            rng.standard_normal((1, e, NUM_EVIDENCE_PLANES, BOARD, BOARD))
        ).astype(np.float32),
        "ev_obs_scalars": rng.standard_normal((1, e, NUM_EVIDENCE_SCALARS), dtype=np.float32),
        "ev_mask": mask,
    }


# One spec per graph: its output head names, its M-riding input names, and the
# per-input dtypes -- everything _probe_graph needs without a family branch.
_INT32 = {"move_letters", "move_squares"}
_UINT8 = {"move_blanks", "move_tile_mask", "ev_mask"}


def _dtype(name: str) -> torch.dtype:
    if name in _INT32:
        return torch.int32
    if name in _UINT8:
        return torch.uint8
    return torch.float32


def _build_refittable(trt, onnx_path: Path, dyn_inputs: dict):
    """Parse + build a kREFIT FP32 engine; `dyn_inputs` maps each M-riding input
    name to its per-row width, for the (1 / OPT / MAX) profile."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}")
        sys.exit(f"GATE FAILED: TensorRT could not parse {onnx_path.name}")
    print("  parse: OK")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT)
    config.builder_optimization_level = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    profile = builder.create_optimization_profile()
    for name, width in dyn_inputs.items():
        profile.set_shape(name, (1, width), (OPT_MOVES, width), (MAX_MOVES, width))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit(f"GATE FAILED: {onnx_path.name} engine build returned None")
    print("  build (kREFIT, dynamic-M profile): OK")
    return trt.Runtime(logger).deserialize_cuda_engine(serialized), logger


def _refit_from(trt, engine, logger, onnx_path: Path):
    refitter = trt.Refitter(engine, logger)
    trt.OnnxParserRefitter(refitter, logger).refit_from_file(str(onnx_path))
    missing = refitter.get_missing_weights()
    if missing:
        print(f"  missing weights after refit: {list(missing)[:10]} ...")
        sys.exit(f"GATE FAILED: {len(missing)} weights unmapped by the refit")
    if not refitter.refit_cuda_engine():
        sys.exit("GATE FAILED: refit_cuda_engine returned false")
    print("  refit with second export's weights: applied (no missing weights)")


def _run_trt(engine, feeds: dict, outputs: dict, dyn_inputs: set) -> dict:
    context = engine.create_execution_context()
    device = {}
    for name, arr in feeds.items():
        device[name] = torch.from_numpy(arr).to("cuda", _dtype(name))
        if name in dyn_inputs:
            context.set_input_shape(name, tuple(arr.shape))
        context.set_tensor_address(name, device[name].data_ptr())
    for name, tensor in outputs.items():
        context.set_tensor_address(name, tensor.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        sys.exit("GATE FAILED: execute_async_v3 returned false")
    torch.cuda.synchronize()
    return {name: tensor.cpu().numpy() for name, tensor in outputs.items()}


def _probe_graph(trt, ort, name, export_a, export_b, make_inputs, out_shapes, dyn_widths):
    """Build+refit+compare one graph. `out_shapes(m)` gives the output buffers;
    `dyn_widths` maps M-riding input names to their widths."""
    print(f"\n[{name}] two random-init exports")
    with tempfile.TemporaryDirectory(prefix=f"proposal_{name}_") as tmp:
        a, b = Path(tmp) / "a.onnx", Path(tmp) / "b.onnx"
        export_a(a)
        export_b(b)
        engine, logger = _build_refittable(trt, a, dyn_widths)
        _refit_from(trt, engine, logger, b)
        sess = ort.InferenceSession(str(b), providers=["CPUExecutionProvider"])
        worst = 0.0
        for m in PROBE_MS:
            feeds = make_inputs(m)
            outputs = {
                oname: torch.empty(shape, dtype=torch.float32, device="cuda")
                for oname, shape in out_shapes(m).items()
            }
            got = _run_trt(engine, feeds, outputs, set(dyn_widths))
            want = sess.run(list(out_shapes(m)), feeds)
            for (oname, arr), w in zip(got.items(), want, strict=True):
                abs_diff = float(np.abs(arr - w).max())
                if oname == "planes":  # unbounded logits: relative error
                    diff = abs_diff / (float(np.abs(w).max()) + 1e-6)
                    tol, unit = PLANES_RTOL, "rel"
                else:
                    diff, tol, unit = abs_diff, TOLERANCE, "abs"
                worst = max(worst, diff)
                status = "OK" if diff < tol else "MISMATCH"
                print(f"  M={m:4d} {oname:10s} {unit}|TRT-ORT| = {diff:.2e}  {status}")
                if diff >= tol:
                    sys.exit(f"GATE FAILED [{name}]: refitted engine disagrees with ORT")
        print(f"  [{name}] worst diff {worst:.2e}")


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    import onnxruntime as ort
    import tensorrt as trt

    print(f"TensorRT {trt.__version__}, torch {torch.__version__}")
    spatial_planes, scalar_size = _shapes()
    t, s, _, _ = move_encoding_dims()
    model_a = _random_model(1, spatial_planes, scalar_size)
    model_b = _random_model(2, spatial_planes, scalar_size)
    channels = model_a.cross_attn.embed_dim
    move_widths = {
        "move_letters": t,
        "move_blanks": t,
        "move_squares": t,
        "move_tile_mask": t,
        "move_scalars": s,
    }

    _probe_graph(
        trt,
        ort,
        "cache",
        lambda p: export_proposal_cache(
            model_a,
            p,
            spatial_planes,
            scalar_size,
            opp_leave_input=False,
            move_encoding_version=1,
            proposal_export_id=proposal_export_id(model_a),
        ),
        lambda p: export_proposal_cache(
            model_b,
            p,
            spatial_planes,
            scalar_size,
            opp_leave_input=False,
            move_encoding_version=1,
            proposal_export_id=proposal_export_id(model_b),
        ),
        lambda m: _cache_inputs(m, spatial_planes, scalar_size, 100 + m),
        lambda m: {
            "board": (1, BOARD * BOARD, channels),
            "g": (1, 3 * channels),
            "move_enc": (m, channels),
            "wld": (m, 3),
            "score_diff": (m, 2),
            "planes": (m, len(PLANE_NAMES), BOARD * BOARD),
        },
        move_widths,
    )
    _probe_graph(
        trt,
        ort,
        "step",
        lambda p: export_proposal_step(
            model_a,
            p,
            opp_leave_input=False,
            move_encoding_version=1,
            proposal_export_id=proposal_export_id(model_a),
        ),
        lambda p: export_proposal_step(
            model_b,
            p,
            opp_leave_input=False,
            move_encoding_version=1,
            proposal_export_id=proposal_export_id(model_b),
        ),
        lambda m: _step_inputs(m, channels, 200 + m),
        lambda m: {
            "wld": (m, 3),
            "score_diff": (m, 2),
            "gain": (m,),
        },
        {"move_enc": channels},
    )
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
