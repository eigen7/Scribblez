#!/usr/bin/env python3
"""The A4 TRT gate: prove the mset ONNX graph parses, builds, and REFITS
correctly under TensorRT before any engine runtime is designed around it
(the reviewed plan's front-loaded check for its #1 risk -- a silently wrong
refit of the embedding tables or the packed MHA in_proj_weight would corrupt
match results while passing every ORT-level parity test).

What it does, on the GPU:
  1. Exports two RANDOMLY-INITIALIZED production-shape models A and B.
  2. Parses A and builds a refittable FP32 engine (optimization level 0; the
     dynamic-M profile the C++ TensorRT runtime will use).
  3. Refits that engine with B's weights through the ONNX parser-refitter,
     asserting no weights go missing. This is the path both runtimes' shared
     plan cache takes for a new checkpoint of a cached architecture
     (engine/src/nn/neural_net.cpp). Step 3's own finding tempers what that
     asserts: TensorRT reports a refit that mapped every weight and one that
     left a weight behind identically, so "no weights go missing" is not the
     assurance it reads as -- which is why step 4's output comparison, not the
     API, is the verification (the C++ parity test does the same per run).
  4. Runs the refitted engine at several Ms and compares against ONNXRuntime
     on B: if the refit silently mapped anything wrong, the outputs are A/B
     chimeras and the comparison fails loudly.

Exit code 0 = gate passed. Uses torch CUDA tensors as TRT buffers, so no
pycuda dependency.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scribblez.ffi import get_input_shapes
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_dims
from scribblez.move_set_eval.onnx_export import MOVE_INPUT_NAMES, export_onnx

MAX_MOVES = 4096
OPT_MOVES = 512
PROBE_MS = (1, 37, 512)
TOLERANCE = 2e-3  # FP32 TRT vs FP32 ORT: kernel-order noise, not semantics


def _shapes():
    dims = {s.name: s.dims for s in get_input_shapes()}
    return dims["input_spatial"][0], dims["input_scalar"][0]


def _export_random(path: Path, seed: int, spatial_planes: int, scalar_size: int):
    torch.manual_seed(seed)
    model = MoveSetEvalModel(spatial_planes, scalar_size)  # production dims
    model.eval()
    export_onnx(
        model,
        path,
        spatial_planes,
        scalar_size,
        opp_leave_input=False,
        move_encoding_version=1,
    )


def _random_inputs(m: int, spatial_planes: int, scalar_size: int, seed: int) -> dict:
    t, s, letter_vocab, cells = move_encoding_dims()
    rng = np.random.default_rng(seed)
    return {
        "input_spatial": rng.standard_normal((1, spatial_planes, 15, 15), dtype=np.float32),
        "input_scalar": rng.standard_normal((1, scalar_size), dtype=np.float32),
        "move_letters": rng.integers(0, letter_vocab, (m, t), dtype=np.int32),
        "move_blanks": rng.integers(0, 2, (m, t)).astype(np.uint8),
        "move_squares": rng.integers(0, cells, (m, t), dtype=np.int32),
        "move_tile_mask": rng.integers(0, 2, (m, t)).astype(np.uint8),
        "move_scalars": rng.standard_normal((m, s)).astype(np.float32),
    }


def _build_refittable_engine(trt, onnx_path: Path):
    """Parse + build with kREFIT and the dynamic-M profile; returns the engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}")
        sys.exit("GATE FAILED: TensorRT could not parse the mset graph")
    print("  parse: OK")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT)
    config.builder_optimization_level = 0  # first-working-kernel; a probe, not production
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    t, s, _, _ = move_encoding_dims()
    profile = builder.create_optimization_profile()
    for name in MOVE_INPUT_NAMES:
        width = s if name == "move_scalars" else t
        profile.set_shape(name, (1, width), (OPT_MOVES, width), (MAX_MOVES, width))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("GATE FAILED: engine build returned None")
    print("  build (kREFIT, M profile 1/512/4096): OK")
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    return engine, logger


def _refit_from(trt, engine, logger, onnx_path: Path):
    refitter = trt.Refitter(engine, logger)
    parser_refitter = trt.OnnxParserRefitter(refitter, logger)
    clean = parser_refitter.refit_from_file(str(onnx_path))
    missing = refitter.get_missing_weights()
    if missing:
        print(f"  missing weights after refit: {list(missing)[:10]} ...")
        sys.exit(f"GATE FAILED: {len(missing)} weights unmapped by the refit")
    if not clean:
        # TRT 10.11's refitOnnxWeights counts one more ONNX-side weight than
        # the engine exposes a slot for (an anonymous fusion product), and
        # fails its own strict count even though the engine reports NO missing
        # weights. The output comparison below is the ground truth: a weight
        # actually left un-refitted keeps the first export's values and cannot
        # reproduce ORT on the second export's.
        print("  refit_from_file: strict count failed, but zero missing weights -- verifying")
    if not refitter.refit_cuda_engine():
        sys.exit("GATE FAILED: refit_cuda_engine returned false")
    print("  refit with second export's weights: applied (no missing weights)")


_TORCH_DTYPES = {
    "input_spatial": torch.float32,
    "input_scalar": torch.float32,
    "move_letters": torch.int32,
    "move_blanks": torch.uint8,
    "move_squares": torch.int32,
    "move_tile_mask": torch.uint8,
    "move_scalars": torch.float32,
}


def _run_trt(engine, feeds: dict, m: int) -> dict:
    """One inference through the (refitted) engine, torch CUDA tensors as
    buffers."""
    context = engine.create_execution_context()
    device_tensors = {}
    for name, arr in feeds.items():
        device_tensors[name] = torch.from_numpy(arr).to("cuda", _TORCH_DTYPES[name])
        if name in MOVE_INPUT_NAMES:
            context.set_input_shape(name, tuple(arr.shape))
        context.set_tensor_address(name, device_tensors[name].data_ptr())
    outputs = {
        "wld": torch.empty(m, 3, dtype=torch.float32, device="cuda"),
        "score_diff": torch.empty(m, 2, dtype=torch.float32, device="cuda"),
    }
    for name, tensor in outputs.items():
        context.set_tensor_address(name, tensor.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        sys.exit("GATE FAILED: execute_async_v3 returned false")
    torch.cuda.synchronize()
    return {name: tensor.cpu().numpy() for name, tensor in outputs.items()}


def main() -> int:
    # No options -- but --help must print this docstring and exit rather than
    # silently running a GPU job on what may be a busy, shared GPU.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    import onnxruntime as ort
    import tensorrt as trt

    print(f"TensorRT {trt.__version__}, torch {torch.__version__}")
    spatial_planes, scalar_size = _shapes()

    with tempfile.TemporaryDirectory(prefix="mset_trt_probe_") as tmp:
        a_path, b_path = Path(tmp) / "a.onnx", Path(tmp) / "b.onnx"
        _export_random(a_path, seed=1, spatial_planes=spatial_planes, scalar_size=scalar_size)
        _export_random(b_path, seed=2, spatial_planes=spatial_planes, scalar_size=scalar_size)
        print("exported two random-init production-shape models")

        engine, logger = _build_refittable_engine(trt, a_path)
        _refit_from(trt, engine, logger, b_path)

        sess = ort.InferenceSession(str(b_path), providers=["CPUExecutionProvider"])
        worst = 0.0
        for m in PROBE_MS:
            feeds = _random_inputs(m, spatial_planes, scalar_size, seed=100 + m)
            got = _run_trt(engine, feeds, m)
            want_wld, want_sd = sess.run(["wld", "score_diff"], feeds)
            for name, want in (("wld", want_wld), ("score_diff", want_sd)):
                diff = float(np.abs(got[name] - want).max())
                worst = max(worst, diff)
                status = "OK" if diff < TOLERANCE else "MISMATCH"
                print(f"  M={m:4d} {name:10s} max|TRT-ORT| = {diff:.2e}  {status}")
                if diff >= TOLERANCE:
                    sys.exit(
                        "GATE FAILED: refitted engine disagrees with ORT on the "
                        "refit weights -- the refit mapped something wrong"
                    )
    print(f"GATE PASSED (worst diff {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
