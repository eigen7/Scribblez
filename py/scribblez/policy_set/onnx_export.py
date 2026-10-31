"""Export the policy-set model's policy form (policy.PolicyModel) to ONNX for
the engine's TensorRT loader (engine/include/nn/model_specs.h,
PolicySetSpec), which PolicyBot plays from.

The graph is specialized to one position (P=1), as the move set export is
(move_set_eval/onnx_export.py describes the specialization and the
TensorRT-refit rebuilds, which apply here too): the board row and the
opponent-rack input, plus M candidates' move features and equity inputs, with
M the only dynamic axis ("moves"). It emits each candidate's logit (M, 1).

The metadata carries what PolicyBot must reproduce of the model's training:
the format and encoding versions, the pool's blank prune (policy_set_inputs.h
pool_candidates), and the leave inference generation whose beliefs the pools'
opponent-rack inputs summarize (empty for uniform beliefs).

The trainer exports every generation into its tag's models/. As a script:
export a tag's rolling checkpoint as its latest generation. The checkpoint
holds the served weights: the trainer checkpoints in eval mode.

    python -m scribblez.policy_set.onnx_export --tag policy-base
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.ffi import task_input_shapes
from scribblez.move_set_eval.moves import move_cross_slots, move_encoding_dims
from scribblez.move_set_eval.onnx_export import (
    MOVE_INPUT_NAMES,
    board_and_summary,
    encode_moves_at_position,
)
from scribblez.onnx_export_util import (
    common_metadata,
    cross_attention_2d,
    export_graph,
    split_concat_linear,
    split_mha_qkv,
)
from scribblez.paths import DEFAULT_MOUNT_ROOT, TagPaths
from scribblez.policy_set import FORM_POLICY
from scribblez.policy_set.dataset import OPP_RACK_FLOATS
from scribblez.policy_set.model import PolicySetEncoder, trunk_scalar
from scribblez.policy_set.policy import PolicyModel

# Must match the engine's nn::kGraphPolicySet.
GRAPH = "policy_set"
OUTPUT = "logit"
WORKLOAD = "policy_set"
# The metadata entries PolicyBot reads (engine agent/policy_bot.h).
BLANK_PRUNE_KEY = "blank_prune_points"
LEAVE_TAG_KEY = "leave_inference_tag"
LEAVE_GENERATION_KEY = "leave_inference_generation"
# The checkpoint-config keys an export needs besides the versions.
SHAPE_KEYS = ("trunk_channels", "num_blocks", "num_heads", "normalize_scalar")


class PolicyExportModel(nn.Module):
    """PolicyModel.forward at P=1, sharing the trained model's submodules.
    Returns the candidates' logits (M, 1)."""

    def __init__(self, model: PolicyModel):
        super().__init__()
        enc = model.encoder
        self.trunk = enc.trunk
        self.board_pos_emb = enc.board_pos_emb
        self.move_encoder = enc.move_encoder
        self.cross_check_encoder = enc.cross_check_encoder
        self.equity = enc.equity
        # Rebuilt for the refitter (move_set_eval/onnx_export.py).
        mha = enc.cross_attn
        c = mha.embed_dim
        self.num_heads = mha.num_heads
        self.q_proj, self.k_proj, self.v_proj = split_mha_qkv(mha)
        self.attn_out = mha.out_proj
        # fuse[0] is Linear(cat([attended (C), g (3C)])); split at C.
        self.fuse_attended, self.fuse_g = split_concat_linear(enc.fuse[0], c)
        self.fuse_out = enc.fuse[2]
        self.vector_norm = enc.vector_norm
        self.logit = model.logit

    def forward(
        self,
        input_spatial: torch.Tensor,  # (1, planes, 15, 15) f32
        input_scalar: torch.Tensor,  # (1, S) f32
        opp_rack: torch.Tensor,  # (1, OPP_RACK_FLOATS) f32
        move_letters: torch.Tensor,  # (M, T) i32
        move_blanks: torch.Tensor,  # (M, T) u8
        move_squares: torch.Tensor,  # (M, T) i32
        move_tile_mask: torch.Tensor,  # (M, T) u8
        move_scalars: torch.Tensor,  # (M, 3) f32
        move_cross_cells: torch.Tensor,  # (M, K) i32
        move_cross_letters: torch.Tensor,  # (M, K * 26) u8
        move_equity: torch.Tensor,  # (M, 1) f32
    ) -> torch.Tensor:
        x, s = self.trunk(input_spatial, trunk_scalar(input_scalar, opp_rack))
        board, g = board_and_summary(self.board_pos_emb, x, s)
        e = encode_moves_at_position(
            self.move_encoder, self.cross_check_encoder, board[0], move_letters, move_blanks,
            move_squares, move_tile_mask, move_scalars, move_cross_cells, move_cross_letters,
        ) + self.equity(move_equity)  # fmt: skip
        attended = cross_attention_2d(
            self.q_proj, self.k_proj, self.v_proj, self.attn_out, self.num_heads, e, board[0]
        )
        h = self.fuse_out(F.relu(self.fuse_attended(attended) + self.fuse_g(g)))
        return self.logit(self.vector_norm(h))


def input_names() -> tuple[str, ...]:
    """The graph's inputs, in order."""
    return ("input_spatial", "input_scalar", "opp_rack", *MOVE_INPUT_NAMES, "move_equity")


def leave_source(params: dict) -> tuple[str, int]:
    """The leave inference tag and generation whose beliefs the pools
    summarize, or ("", -1) for uniform beliefs."""
    if params["leave_beliefs"] != "inferred":
        return "", -1
    return params["leave_inference_tag"], int(params["leave_inference_generation"])


def export_onnx(
    model: PolicyModel,
    path: str | Path,
    params: dict,
    provenance: dict[str, object],
    *,
    board_size: int = 15,
    opset: int = 17,
):
    """Export `model` to `path`, atomically. `params` is the tag's task
    params (the checkpoint config); `provenance` (tag, generation) is
    recorded verbatim. The metadata records the checkpoint's versions, which
    the engine refuses unless they are its own."""
    device = next(model.parameters()).device
    wrapper = PolicyExportModel(model).to(device)
    shapes = {s.name: s.dims for s in task_input_shapes()}
    t, num_scalars, _, _ = move_encoding_dims()
    k = move_cross_slots()
    dummy_m = 5  # any M > 1; export_graph's check runs another
    dummies = (
        torch.zeros(1, shapes["input_spatial"][0], board_size, board_size, device=device),
        torch.zeros(1, shapes["input_scalar"][0], device=device),
        torch.zeros(1, OPP_RACK_FLOATS, device=device),
        torch.zeros(dummy_m, t, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, t, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, t, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, t, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, num_scalars, device=device),
        torch.zeros(dummy_m, k, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, k * 26, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, 1, device=device),
    )
    names = input_names()
    leave_tag, leave_generation = leave_source(params)
    export_graph(
        wrapper,
        dummies,
        path,
        input_names=names,
        output_names=(OUTPUT,),
        dynamic_axes={name: {0: "moves"} for name in (*names[3:], OUTPUT)},
        metadata={
            **common_metadata(opp_leave_input=False),
            "graph": GRAPH,
            # The checkpoint's versions, which the engine checks against its own.
            "input_encoding_version": str(params["input_encoding_version"]),
            "policy_set_format_version": str(params["policy_set_format_version"]),
            "move_encoding_version": str(params["move_encoding_version"]),
            BLANK_PRUNE_KEY: str(params["blank_prune_points"]),
            LEAVE_TAG_KEY: leave_tag,
            LEAVE_GENERATION_KEY: str(leave_generation),
            **{key: str(v) for key, v in provenance.items()},
        },
        opset=opset,
    )


def policy_model(ckpt: dict) -> PolicyModel:
    """The policy form of the loaded rolling checkpoint `ckpt`, in eval mode."""
    params = ckpt["config"]
    if not params["train_policy"]:
        raise ValueError("the checkpoint trains no policy form")
    shapes = {s.name: s.dims for s in task_input_shapes()}
    model = PolicyModel(
        PolicySetEncoder(
            shapes["input_spatial"][0],
            shapes["input_scalar"][0],
            *(params[key] for key in SHAPE_KEYS),
        )
    )
    prefix = f"{FORM_POLICY}."
    model.load_state_dict(
        {k[len(prefix) :]: v for k, v in ckpt["model_state_dict"].items() if k.startswith(prefix)}
    )
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help=f"a {WORKLOAD} tag name")
    parser.add_argument(
        "--out", help="output .onnx (default: the tag's export of that generation, in models/)"
    )
    args = parser.parse_args()
    paths = TagPaths(args.tag, WORKLOAD, DEFAULT_MOUNT_ROOT)
    if not paths.rolling_checkpoint.exists():
        parser.error(f"tag {args.tag} has no checkpoint at {paths.rolling_checkpoint}")
    ckpt = torch.load(paths.rolling_checkpoint, map_location="cpu", weights_only=False)
    if not ckpt["config"]["train_policy"]:
        parser.error(f"tag {args.tag} trains no policy form")
    # generation_index is the next generation to train; the weights are the
    # previous one's.
    generation = int(ckpt["generation_index"]) - 1
    out = Path(args.out) if args.out else paths.onnx_path(generation)
    export_onnx(
        policy_model(ckpt), out, ckpt["config"], {"tag": args.tag, "generation": generation}
    )
    print(f"exported {out} (generation {generation})")


if __name__ == "__main__":
    main()
