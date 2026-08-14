#!/usr/bin/env python3
"""Generate a fixture for the C++ move-set inference parity test (docs/roadmap.md, A4).

The agent will run the move set evaluation model through onnx_export ->
TensorRT, while training and the dashboard run the in-memory PyTorch model.
This script captures the PyTorch side as ground truth so the C++ test can
confirm the TensorRT path -- engine build, the dtype-aware bindings, the
chunking, and the C++ Eval decode -- reproduces it.

Two models of the same architecture are exported to prove the engine-plan cache
never confuses them. The cache is keyed on the model's exact content, so the
second one must build and cache its own plan rather than reuse the first's; its
own reference outputs are what shows which weights were actually served, a
question no shape or metadata check can answer, since two checkpoints of one
architecture differ in nothing but their numbers.

Files written into --out-dir:
  * model_a.onnx / model_b.onnx -- randomly-initialized MoveSetEvalModels of one
    architecture, exported at the current move-encoding version.
  * model_stale.onnx -- model A stamped with move-encoding version 0, for the
    version-skew guard.
  * model_narrow.onnx / model_uint8_letters.onnx -- model A with one fewer tile
    slot per move, and model A declaring move_letters as uint8, for the two
    halves of the engine's move-feature layout guard (row width and element
    size). Both still build under TensorRT: the point is a model the engine must
    refuse to feed, not one it cannot parse.
  * board.bin -- one position's encoder row (spatial floats then scalar floats),
    float32, laid out as GameStateEncoder::encode_input writes it.
  * move_{letters,blanks,squares,tile_mask,scalars}.bin -- the candidate set, in
    move_set_encoder.h's own dtypes and row-major layout. M is recovered C++-side
    from the scalars file's size.
  * expected_a.bin / expected_b.bin -- M x 6 float32: [win_prob, p_win, p_draw,
    p_loss, score_diff_mean, score_diff_std], each model's PyTorch decode.

Random weights are deliberate: this checks numerical fidelity of the inference
stack, not the quality of any trained model, and keeps the fixture hermetic.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from scribblez.ffi import get_input_shapes, set_contingent_features
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.moves import move_encoding_dims, move_encoding_version
from scribblez.move_set_eval.onnx_export import export_onnx

# A deliberately tiny architecture: the parity check exercises the inference
# stack, and a small trunk keeps the TensorRT builds to seconds. num_blocks=3
# includes one global-pooling block, so the global summary path is covered.
TRUNK_CHANNELS = 8
NUM_BLOCKS = 3
NUM_HEADS = 2

# The board layout is owned by the C++ encoder and surfaced through the FFI, so
# the fixture's row always matches the widths the C++ test reads back off the
# loaded model.
CONTINGENT_FEATURES = True


def input_widths() -> tuple[int, int, int]:
    set_contingent_features(CONTINGENT_FEATURES)
    dims = {s.name: s.dims for s in get_input_shapes()}
    planes, height, width = dims["input_spatial"]
    assert height == width, "the model assumes a square board"
    return planes, height, dims["input_scalar"][0]


def build_model(seed: int, spatial_planes: int, scalar_size: int, board_size: int):
    torch.manual_seed(seed)
    model = MoveSetEvalModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        trunk_channels=TRUNK_CHANNELS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        board_size=board_size,
    )
    model.eval()
    return model


def random_candidates(num_moves: int, seed: int) -> dict[str, np.ndarray]:
    """A candidate set shaped the way move_set_encoder.h fills one: each move
    carries 1..7 tiles, and a fifth of them are exchanges, which carry letters
    and blanks but no squares (the is_play gate's whole subject)."""
    tiles, num_scalars, letter_vocab, cells = move_encoding_dims()
    rng = np.random.default_rng(seed)

    letters = np.zeros((num_moves, tiles), dtype=np.int32)
    blanks = np.zeros((num_moves, tiles), dtype=np.uint8)
    squares = np.zeros((num_moves, tiles), dtype=np.int32)
    tile_mask = np.zeros((num_moves, tiles), dtype=np.uint8)
    scalars = np.zeros((num_moves, num_scalars), dtype=np.float32)

    for m in range(num_moves):
        is_play = float(m % 5 != 0)
        n = int(rng.integers(1, tiles + 1))
        letters[m, :n] = rng.integers(1, letter_vocab, n)
        blanks[m, :n] = rng.integers(0, 2, n)
        tile_mask[m, :n] = 1
        if is_play:
            squares[m, :n] = rng.integers(0, cells, n)
        scalars[m] = [rng.standard_normal(), n / tiles, is_play]

    return {
        "move_letters": letters,
        "move_blanks": blanks,
        "move_squares": squares,
        "move_tile_mask": tile_mask,
        "move_scalars": scalars,
    }


@torch.no_grad()
def reference_evals(model, board_row: np.ndarray, moves: dict, shape: tuple[int, int, int]):
    """The training model's own forward over the candidate set, decoded into
    [win_prob, p_win, p_draw, p_loss, sd_mean, sd_std] per move.

    The reference is `MoveSetEvalModel.forward` rather than the export wrapper,
    so this fixture measures the whole chain the agent will run against what
    training actually optimized; the wrapper's equivalence to it is the Python
    parity suite's job (py/tests/test_move_set_inference_parity.py).
    """
    spatial_planes, board_size, scalar_size = shape
    spatial_floats = spatial_planes * board_size * board_size
    spatial = torch.from_numpy(board_row[:spatial_floats]).reshape(
        1, spatial_planes, board_size, board_size
    )
    scalar = torch.from_numpy(board_row[spatial_floats:]).reshape(1, scalar_size)

    num_moves = moves["move_scalars"].shape[0]
    out = model(
        spatial,
        scalar,
        torch.from_numpy(moves["move_letters"]).long(),
        torch.from_numpy(moves["move_blanks"]),
        torch.from_numpy(moves["move_squares"]).long(),
        torch.from_numpy(moves["move_tile_mask"]).float(),
        torch.from_numpy(moves["move_scalars"]),
        torch.zeros(num_moves, dtype=torch.long),  # the single position
    )

    wld = torch.softmax(out["wld"], dim=1).numpy()  # [win, draw, loss]
    p_win, p_draw, p_loss = wld[:, 0], wld[:, 1], wld[:, 2]
    win_prob = p_win + 0.5 * p_draw
    sd = out["score_diff"].numpy()  # already [mean, std>0]
    return np.stack([win_prob, p_win, p_draw, p_loss, sd[:, 0], sd[:, 1]], axis=1).astype(
        np.float32
    )


def write_narrowed(src: Path, dst: Path):
    """Copy `src` with one fewer tile slot per move, for the engine's
    move-feature width guard.

    Every per-tile input is narrowed together, because the move encoder adds
    their embeddings elementwise: narrowing one alone would be a graph
    TensorRT refuses to build, and the guard is meant to catch a model that
    builds fine and is simply not the one this encoder feeds. Nothing else in
    the graph fixes the tile count -- the pooling is a sum over that axis -- so
    the result is a valid model of a slightly different encoding.
    """
    per_tile = {"move_letters", "move_blanks", "move_squares", "move_tile_mask"}
    model = onnx.load(src)
    for tensor in model.graph.input:
        if tensor.name in per_tile:
            slot = tensor.type.tensor_type.shape.dim[1]
            slot.dim_value = slot.dim_value - 1
    onnx.save(model, dst)


def write_uint8_letters(src: Path, dst: Path):
    """Copy `src` declaring move_letters as uint8, for the element-size half of
    the same guard.

    A letter fits in a byte, so this is the plausible version of the mistake --
    and the graph is still valid, because the first thing done with the input
    is a cast. Only the engine's staging buffer, sized from the declaration and
    written at the encoder's int32, would notice.
    """
    model = onnx.load(src)
    for tensor in model.graph.input:
        if tensor.name == "move_letters":
            tensor.type.tensor_type.elem_type = onnx.TensorProto.UINT8
    onnx.save(model, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-moves", type=int, default=37)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spatial_planes, board_size, scalar_size = input_widths()
    shape = (spatial_planes, board_size, scalar_size)
    version = move_encoding_version()

    def export(model, name: str, encoding_version: int):
        export_onnx(
            model,
            args.out_dir / name,
            spatial_planes,
            scalar_size,
            contingent_features=CONTINGENT_FEATURES,
            opp_leave_input=False,
            move_encoding_version=encoding_version,
            board_size=board_size,
        )

    model_a = build_model(args.seed, spatial_planes, scalar_size, board_size)
    model_b = build_model(args.seed + 1, spatial_planes, scalar_size, board_size)
    export(model_a, "model_a.onnx", version)
    export(model_b, "model_b.onnx", version)
    export(model_a, "model_stale.onnx", 0)
    write_narrowed(args.out_dir / "model_a.onnx", args.out_dir / "model_narrow.onnx")
    write_uint8_letters(args.out_dir / "model_a.onnx", args.out_dir / "model_uint8_letters.onnx")

    rng = np.random.default_rng(args.seed)
    board_row = rng.standard_normal(
        spatial_planes * board_size * board_size + scalar_size, dtype=np.float32
    )
    moves = random_candidates(args.num_moves, args.seed)

    (args.out_dir / "board.bin").write_bytes(board_row.tobytes())
    for name, array in moves.items():
        (args.out_dir / f"{name}.bin").write_bytes(array.tobytes())
    for tag, model in (("a", model_a), ("b", model_b)):
        expected = reference_evals(model, board_row, moves, shape)
        (args.out_dir / f"expected_{tag}.bin").write_bytes(expected.tobytes())

    print(f"Wrote fixture to {args.out_dir}:")
    print(f"  models a/b/stale (move-encoding version {version}, stale 0)")
    print(f"  board.bin ({board_row.size} floats), {args.num_moves} candidate moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
