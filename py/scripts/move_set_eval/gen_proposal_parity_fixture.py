#!/usr/bin/env python3
"""Generate a fixture for the C++ move-proposal inference parity test (roadmap item 3).

The engine runs the move proposal model as two graphs -- a per-turn
`move_proposal_cache` and a per-evidence-iteration `move_proposal_step`
(py/scribblez/move_set_eval/proposal_export.py) -- driven by
agent/move_proposal_runtime.h and staged by agent/evidence_staging.h. This
script captures the PyTorch side as ground truth so the C++ test can confirm the
whole path -- two engine builds, the board/g/move_enc host handoff, the C++
evidence staging, and the decode -- reproduces `MoveSetEvalModel.forward`.

The reference is `MoveSetEvalModel.forward(evidence=...)` where the evidence set
is assembled by the SAME builder (evidence.py's build_evidence_inputs) that the
C++ evidence_staging.cpp is a port of, from the SAME raw sim observations the C++
test is handed. So the two sides see identical evidence and the comparison is a
true engine-vs-PyTorch parity check, tolerance-bounded (independent TensorRT
plans are not bit-identical to PyTorch).

Key invariant: every candidate is encoded with ONE pre-move differential, so the
runtime's gather of move_enc[scored_index] equals the reference's re-encode of
that evidence candidate. The evidence cases deliberately use scattered and
duplicate indices so a gather/routing swap shows as a mismatch rather than
aliasing away.

Files written into --out-dir:
  * cache.onnx / step.onnx -- the split graphs of one randomly-initialized model
    (its evidence path perturbed off zero-init, so a populated set differs from
    the plain pass), exported at the current move-encoding version and E=MAX_E.
  * step_mismatch.onnx -- a second model's step graph (same architecture,
    different weights, so a different proposal_export_id), for the runtime's
    cache/step pairing guard.
  * board.bin -- one position's encoder row (spatial then scalar floats), f32.
  * move_{letters,blanks,squares,tile_mask,scalars}.bin -- the M candidates in
    move_set_encoder.h's dtypes, encoded with the single pre-move differential.
  * moves_sobs.bin -- the M candidates as 16-byte Move records (MOVE_DTYPE), for
    the C++ Move footprint; obs.bin -- their SimObservation records (verbatim
    layout), for the C++ SimObservation. M is recovered C++-side from board /
    scalars sizes and obs.bin.
  * cases.txt -- one line per evidence case: "<name> <num_evidence>".
  * case_<name>_indices.bin -- int32 scored indices (empty file for the empty
    case); case_<name>_scalars.bin -- M x 6 f32 [p_win, p_draw, p_loss, sd_mean,
    sd_std, gain]; case_<name>_planes.bin -- M x 52 x 225 f32 footprint
    slot-channel planes (what the graphs serve).

Random weights are deliberate: this checks numerical fidelity of the inference
stack, not any trained model, and keeps the fixture hermetic.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scribblez.ffi import encode_moves, get_input_shapes
from scribblez.footprint_spatial import NUM_CLASSES
from scribblez.move_set_eval.evidence import build_evidence_inputs
from scribblez.move_set_eval.model import MoveSetEvalModel, footprint_slot_planes
from scribblez.move_set_eval.moves import move_encoding_version
from scribblez.move_set_eval.proposal_export import (
    DEFAULT_MAX_EVIDENCE,
    export_proposal_cache,
    export_proposal_step,
    proposal_export_id,
)
from scribblez.sim_evidence.sobs import BOARD, MOVE_DTYPE, MOVE_EXCHANGE, MOVE_PLAY, RECORD_DTYPE

# A deliberately tiny architecture, as the mset parity fixture uses: the check
# exercises the inference stack, and a small trunk keeps the TensorRT builds to
# seconds. num_blocks=3 includes one global-pooling block.
TRUNK_CHANNELS = 8
NUM_BLOCKS = 3
NUM_HEADS = 2

# The step graph's padded evidence width -- must equal the engine's
# nn::kMaxEvidence, so the exported ev_* inputs are the width the step spec's
# layout check expects.
MAX_E = DEFAULT_MAX_EVIDENCE

# The single pre-move differential every candidate is encoded with (see the
# gather==re-encode invariant in the module docstring).
PRE_MOVE_DIFF = 30

OBS_DTYPE = RECORD_DTYPE["obs"]


def input_widths() -> tuple[int, int, int]:
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
    # The fusion projections and proves-best head are zero-init, which would make
    # every conditioned pass equal the plain one -- perturb them so a populated
    # evidence set genuinely exercises the fusion + gain path.
    with torch.no_grad():
        for module in (model.evidence_fusion, model.proves_best):
            for p in module.parameters():
                p.add_(0.1 * torch.randn_like(p))
    model.eval()
    return model


def synthetic_moves(num_moves: int, seed: int) -> np.ndarray:
    """M Move records (MOVE_DTYPE): mostly plays of 1..5 tiles with varied
    footprints, and every seventh an exchange (no squares) -- exercising the
    is_play gate and empty footprints."""
    rng = np.random.default_rng(seed)
    moves = np.zeros(num_moves, dtype=MOVE_DTYPE)
    for i in range(num_moves):
        move = moves[i]
        if i % 7 == 6:  # an exchange: tiles surrendered, no board squares
            move["type"] = MOVE_EXCHANGE
            n = int(rng.integers(1, 6))
            move["num_played"] = n
            move["glyphs"][:n] = rng.integers(1, 27, n)
            continue
        move["type"] = MOVE_PLAY
        move["horizontal"] = int(i % 2 == 0)
        move["start"] = int(rng.integers(0, BOARD))
        n = int(rng.integers(1, 6))
        move["num_played"] = n
        move["glyphs"][:n] = rng.integers(1, 27, n)
        along = int(rng.integers(0, BOARD - n + 1))
        move["square_mask"] = sum(1 << (along + t) for t in range(n))
        move["score"] = int(10 + i)
    return moves


def synthetic_observations(num_moves: int, seed: int) -> np.ndarray:
    """M SimObservation records with plausible rollout aggregates -- distinct
    per candidate so a mis-gather is visible."""
    rng = np.random.default_rng(seed + 1)
    obs = np.zeros(num_moves, dtype=OBS_DTYPE)
    for i in range(num_moves):
        o = obs[i]
        n = int(rng.integers(20, 120))
        o["n"] = n
        wins = int(rng.integers(0, n + 1))
        draws = int(rng.integers(0, n - wins + 1))
        o["wins"], o["draws"], o["losses"] = wins, draws, n - wins - draws
        mean = float(rng.integers(-40, 41))
        o["delta_sum"] = mean * n
        o["delta_sq_sum"] = (mean**2 + float(rng.integers(0, 60))) * n
        o["opp_next_count"][:] = rng.integers(0, n + 1, size=NUM_CLASSES).astype(np.uint16)
        o["self_next_count"][:] = rng.integers(0, n + 1, size=NUM_CLASSES).astype(np.uint16)
        o["opp_win_count"][:] = rng.random(NUM_CLASSES).astype(np.float32) * n
        o["self_win_count"][:] = rng.random(NUM_CLASSES).astype(np.float32) * n
    return obs


def evidence_indices(num_moves: int) -> list[tuple[str, list[int]]]:
    """The evidence cases: empty, a partial (k<E) set with scattered indices and
    a duplicate, and a full (k==E) set -- exercising the padding boundary at both
    ends plus the scattered/duplicate gather."""
    rng = np.random.default_rng(0)
    partial = rng.permutation(num_moves)[:5].tolist()
    partial[1] = partial[0]  # the same simmed candidate appearing twice
    full = rng.permutation(num_moves)[:MAX_E].tolist()
    return [("empty", []), ("partial", partial), ("full", full)]


@torch.no_grad()
def forward(model, spatial, scalar, enc, num_moves: int, evidence=None) -> dict[str, torch.Tensor]:
    """MoveSetEvalModel.forward over the encoded candidate set -- the plain pass
    with `evidence` None, the conditioned pass with an EvidenceInputs."""
    return model(
        spatial,
        scalar,
        torch.from_numpy(enc["letters"]).long(),
        torch.from_numpy(enc["blanks"]).bool(),
        torch.from_numpy(enc["squares"]).long(),
        torch.from_numpy(enc["tile_mask"]).float(),
        torch.from_numpy(enc["scalars"]),
        torch.zeros(num_moves, dtype=torch.long),
        evidence=evidence,
    )


def decode(out: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """forward() outputs -> (scalars M x 6 [p_win, p_draw, p_loss, sd_mean,
    sd_std, gain], planes M x 52 x 225 -- the footprint heads' slot-channel
    probabilities, what the proposal graphs serve)."""
    wld = torch.softmax(out["wld"], dim=1).numpy()
    sd = out["score_diff"].numpy()
    gain = out["gain"].numpy()
    scalars = np.concatenate([wld, sd, gain[:, None]], axis=1).astype(np.float32)
    planes = footprint_slot_planes(out["planes"]).flatten(2).numpy().astype(np.float32)
    return scalars, planes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-moves", type=int, default=70)  # >= MAX_E, for the full case
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert args.num_moves >= MAX_E, "the full evidence case needs num_moves >= MAX_E"

    spatial_planes, board_size, scalar_size = input_widths()
    version = move_encoding_version()
    model = build_model(args.seed, spatial_planes, scalar_size, board_size)

    xid = proposal_export_id(model)
    export_proposal_cache(
        model,
        args.out_dir / "cache.onnx",
        spatial_planes,
        scalar_size,
        opp_leave_input=False,
        move_encoding_version=version,
        proposal_export_id=xid,
        board_size=board_size,
    )
    export_proposal_step(
        model,
        args.out_dir / "step.onnx",
        opp_leave_input=False,
        move_encoding_version=version,
        proposal_export_id=xid,
        max_evidence=MAX_E,
        board_size=board_size,
    )
    # A second model's step graph -- same architecture, different weights, hence
    # a different proposal_export_id -- for the runtime's cache/step pairing
    # guard (loading cache.onnx with this must be rejected).
    alt = build_model(args.seed + 1, spatial_planes, scalar_size, board_size)
    export_proposal_step(
        alt,
        args.out_dir / "step_mismatch.onnx",
        opp_leave_input=False,
        move_encoding_version=version,
        proposal_export_id=proposal_export_id(alt),
        max_evidence=MAX_E,
        board_size=board_size,
    )

    rng = np.random.default_rng(args.seed)
    board_row = rng.standard_normal(
        spatial_planes * board_size * board_size + scalar_size, dtype=np.float32
    )
    spatial = torch.from_numpy(board_row[: spatial_planes * board_size * board_size]).reshape(
        1, spatial_planes, board_size, board_size
    )
    scalar = torch.from_numpy(board_row[spatial_planes * board_size * board_size :]).reshape(
        1, scalar_size
    )

    moves = synthetic_moves(args.num_moves, args.seed)
    obs = synthetic_observations(args.num_moves, args.seed)
    pre_diffs = np.full(args.num_moves, PRE_MOVE_DIFF, dtype=np.int32)
    enc = encode_moves(moves, pre_diffs)
    # The cache graph's inputs, in move_set_encoder.h's own dtypes.
    cache_inputs = {
        "move_letters": enc["letters"].astype(np.int32),
        "move_blanks": enc["blanks"].astype(np.uint8),
        "move_squares": enc["squares"].astype(np.int32),
        "move_tile_mask": enc["tile_mask"].astype(np.uint8),
        "move_scalars": enc["scalars"].astype(np.float32),
    }

    first_pass = forward(model, spatial, scalar, enc, args.num_moves)
    fp = {k: first_pass[k] for k in ("wld", "score_diff", "planes")}

    (args.out_dir / "board.bin").write_bytes(board_row.tobytes())
    for name, array in cache_inputs.items():
        (args.out_dir / f"{name}.bin").write_bytes(np.ascontiguousarray(array).tobytes())
    (args.out_dir / "moves_sobs.bin").write_bytes(np.ascontiguousarray(moves).tobytes())
    (args.out_dir / "obs.bin").write_bytes(np.ascontiguousarray(obs).tobytes())

    cases = evidence_indices(args.num_moves)
    with (args.out_dir / "cases.txt").open("w") as f:
        for name, indices in cases:
            f.write(f"{name} {len(indices)}\n")

    for name, indices in cases:
        k = len(indices)
        if k == 0:
            evidence = build_evidence_inputs(
                moves[:0], obs[:0], PRE_MOVE_DIFF, {key: fp[key][:0] for key in fp}, max_e=MAX_E
            )
        else:
            idx = np.asarray(indices, dtype=np.int64)
            evidence = build_evidence_inputs(
                moves[idx], obs[idx], PRE_MOVE_DIFF, {key: fp[key][idx] for key in fp}, max_e=MAX_E
            )
        out = forward(model, spatial, scalar, enc, args.num_moves, evidence)
        scalars, planes = decode(out)
        (args.out_dir / f"case_{name}_indices.bin").write_bytes(
            np.asarray(indices, dtype=np.int32).tobytes()
        )
        (args.out_dir / f"case_{name}_scalars.bin").write_bytes(scalars.tobytes())
        (args.out_dir / f"case_{name}_planes.bin").write_bytes(planes.tobytes())

    print(f"Wrote proposal parity fixture to {args.out_dir}:")
    print(f"  cache/step graphs (move-encoding version {version}, E={MAX_E})")
    print(f"  {args.num_moves} candidates, cases: {', '.join(n for n, _ in cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
