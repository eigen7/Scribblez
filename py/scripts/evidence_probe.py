#!/usr/bin/env python3
"""Single-position sim-evidence probe (docs/sim_residual_feedback.md).

Evaluates kill-test-trained models on one penultimate-bingo analysis GCG
(the position-eval test dataset format), with and without the position's
sim evidence, to make the evidence's effect inspectable head by head:

  - runs SimRunner live on the GCG's final decision point (HastyBot-equity
    top-K candidates, common-random-number rollouts);
  - prints the per-candidate sim results, the sim's hottest opponent-reply /
    opponent-danger squares, and the model's own predicted danger squares --
    the spatial residual ("this spot is hotter than you think") in raw form;
  - prints each arm's WLD and score-diff heads with evidence zeroed vs.
    supplied, next to the position's committed Monte-Carlo ground truth.

Requires the arms' saved weights (<mount>/tags/kill_test/<tag>/cache/results/
<arm>_model.pt), which kill_test.py writes during training.

Usage:
    ./py/scripts/evidence_probe.py -t apple \\
        --gcg positions/NWL23/position-eval-test-dataset/pos-6.gcg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scribblez.ffi import (
    analyze_post_move_gcg,
    gcg_sim_evidence,
    get_input_shapes,
    set_opp_leave_input,
)
from scribblez.position_eval.analysis import ground_truth_path
from scribblez.sim_evidence.model import EvidencePositionEvalModel
from scribblez.sim_evidence.sobs import (
    MOVE_EXCHANGE,
    MOVE_PLAY,
    SobsPosition,
    evidence_features,
    glyph_char,
    move_footprint,
)
from scribblez.workloads import kill_test as kill_test_workload


def square_name(r: int, c: int) -> str:
    """Column-letter + 1-based-row coordinate, e.g. (0, 14) -> 'O1' (the web
    UI's convention)."""
    return f"{chr(ord('A') + c)}{r + 1}"


def describe_move(move: np.void) -> str:
    tiles = "".join(glyph_char(int(g)) for g in move["glyphs"][: int(move["num_played"])])
    if int(move["type"]) == MOVE_EXCHANGE:
        return f"EXCHANGE {tiles}"
    if int(move["type"]) != MOVE_PLAY:
        return "PASS"
    foot = move_footprint(move)
    rr, cc = np.nonzero(foot)
    anchor = square_name(int(rr[0]), int(cc[0]))
    direction = "across" if move["horizontal"] else "down"
    return f"PLAY {tiles} @ {anchor} {direction} ({int(move['score'])} pts)"


def top_squares(plane: np.ndarray, k: int = 6, min_value: float = 0.01) -> str:
    flat = plane.flatten()
    order = np.argsort(flat)[::-1][:k]
    parts = [f"{square_name(i // 15, i % 15)}={flat[i]:.2f}" for i in order if flat[i] >= min_value]
    return "  ".join(parts) if parts else "(none)"


def load_arm(results_dir: Path, arm: str, suffix: str, open_leaves: bool, device):
    """Rebuild an arm's model from its saved weights and recorded args."""
    meta = json.loads((results_dir / f"{arm}{suffix}.json").read_text())
    arm_open = bool(meta["args"].get("open_leaves", False))
    if arm_open != open_leaves:
        raise SystemExit(
            f"arm `{arm}{suffix}` was trained with open_leaves={arm_open}; rerun the probe "
            f"with{'' if arm_open else 'out'} --open-leaves"
        )
    weights = results_dir / f"{arm}{suffix}_model.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} missing -- rerun kill_test.py once to save arm weights")
    shapes = {s.name: s.dims for s in get_input_shapes()}
    model = EvidencePositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=meta["args"]["trunk_channels"],
        num_blocks=meta["args"]["num_blocks"],
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model, meta["args"]["max_k"]


@torch.no_grad()
def run_heads(model, spatial, scalar, planes, scalars, mask) -> dict:
    out = model(spatial, scalar, planes, scalars, mask)
    wld = out["wld"].softmax(dim=1)[0].tolist()
    return {
        "win": wld[0],
        "draw": wld[1],
        "loss": wld[2],
        "sd_mean": float(out["score_diff"][0, 0]),
        "sd_std": float(out["score_diff"][0, 1]),
        "opp_win_map": torch.sigmoid(out["opp_win_placement"][0]).cpu().numpy(),
    }


def mc_ground_truth(gcg_path: Path, open_leaves: bool) -> dict | None:
    """The position's committed Monte-Carlo truth under the probe's information
    condition, or None when the dataset has none."""
    results = ground_truth_path(gcg_path.parent, open_leaves)
    if not results.exists():
        return None
    entry = json.loads(results.read_text()).get(gcg_path.stem)
    if entry is None:
        return None
    n = entry["n"]
    wld = entry["wld"]
    deltas = {int(d): c for d, c in entry["score_delta_hist"].items()}
    mean = sum(d * c for d, c in deltas.items()) / n
    return {"win": wld["win"] / n, "draw": wld["draw"] / n, "loss": wld["loss"] / n, "mean": mean}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-t", "--tag", required=True, help="kill_test.py tag whose arms to load")
    p.add_argument("--gcg", required=True, help="penultimate-bingo analysis GCG")
    p.add_argument("--arms", default="none,scalar,full", help="comma-separated arms to evaluate")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--rollouts",
        type=int,
        default=200,
        help="rollouts per candidate; the default matches the training evidence so the "
        "model sees in-distribution noise levels (raise it for a sharper sim table)",
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--suffix",
        default="",
        help="result-file suffix of the arms to load (kill_test.py --out-suffix)",
    )
    p.add_argument(
        "--open-leaves",
        action="store_true",
        help="the open-leaves information condition: encode the input with the "
        "opponent-leave block and sim with their retained leave known; the arms "
        "must have been trained with kill_test.py --open-leaves. (On the "
        "penultimate-bingo datasets the to-act player just bingoed, so the known "
        "leave is empty and the sims coincide with hidden mode -- expected.)",
    )
    p.add_argument(
        "--include-own-sim",
        action="store_true",
        help="keep the played move's own sim in the evidence (the arms train "
        "leave-one-out by default, so the default here matches training)",
    )
    args = p.parse_args()
    set_opp_leave_input(args.open_leaves)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcg_path = Path(args.gcg)
    gcg_text = gcg_path.read_text()
    results_dir = kill_test_workload.SPEC.data_dir(args.tag) / "cache" / "results"

    # --- live sim evidence at the decision point ---
    records, played_rank = gcg_sim_evidence(
        gcg_text,
        top_k=args.top_k,
        rollouts=args.rollouts,
        threads=args.threads,
        seed=args.seed,
        open_leaves=args.open_leaves,
    )
    print(
        f"=== sim evidence: {len(records)} candidates x {args.rollouts} rollouts "
        f"(played move rank: {played_rank}) ==="
    )
    for i, rec in enumerate(records):
        obs = rec["obs"]
        n = max(int(obs["n"]), 1)
        marker = " <= played" if i == played_rank else ""
        print(
            f"  [{i}] {describe_move(rec['move']):42s} "
            f"win={obs['wins'] / n:.3f} draw={obs['draws'] / n:.3f} "
            f"delta={float(obs['delta_sum']) / n:+7.1f}{marker}"
        )

    # The played move's rollouts describe the future the model is evaluating.
    ref = records[played_rank if played_rank >= 0 else 0]["obs"]
    n = max(int(ref["n"]), 1)
    opp_next = ref["opp_next_count"].astype(np.float32).reshape(15, 15) / n
    opp_win = ref["opp_win_count"].astype(np.float32).reshape(15, 15) / n
    print("\n=== sim: opponent reply hot spots (fraction of rollouts) ===")
    print(f"  opp plays here:          {top_squares(opp_next)}")
    print(f"  opp plays here AND wins: {top_squares(opp_win)}")

    # --- model heads, with and without the evidence ---
    row = analyze_post_move_gcg(gcg_text)
    shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_floats = int(np.prod(shapes["input_spatial"]))
    spatial = torch.from_numpy(row[:spatial_floats].reshape(1, *shapes["input_spatial"])).to(device)
    scalar = torch.from_numpy(row[spatial_floats:].reshape(1, -1)).to(device)

    truth = mc_ground_truth(gcg_path, args.open_leaves)
    if truth:
        print("\n=== Monte-Carlo ground truth (committed, 10k rollouts) ===")
        print(
            f"  win={truth['win']:.3f} draw={truth['draw']:.3f} loss={truth['loss']:.3f} "
            f"delta={truth['mean']:+.1f}"
        )

    print("\n=== model heads (win/draw/loss, score-diff mean+-std) ===")
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        model, max_k = load_arm(results_dir, arm, args.suffix, args.open_leaves, device)
        pos = SobsPosition(
            game_index=0,
            turn_index=0,
            rollouts=args.rollouts,
            base_seed=args.seed,
            num_legal_moves=0,
            flags=0,
            moves=records["move"],
            obs=records["obs"],
        )
        planes, scalars, mask = evidence_features(pos, max_k)
        # The arms train leave-one-out, so mask the played move's own sim to
        # keep the probe's evidence in-distribution.
        if not args.include_own_sim and played_rank >= 0 and played_rank < max_k:
            planes[played_rank] = 0
            scalars[played_rank] = 0
            mask[played_rank] = False
        ev = (
            torch.from_numpy(planes).unsqueeze(0).float().to(device),
            torch.from_numpy(scalars).unsqueeze(0).to(device),
            torch.from_numpy(mask).unsqueeze(0).to(device),
        )
        zero = (torch.zeros_like(ev[0]), torch.zeros_like(ev[1]), torch.zeros_like(ev[2]))

        danger = {}
        for label, inputs in (("no evidence", zero), ("with evidence", ev)):
            h = run_heads(model, spatial, scalar, *inputs)
            print(
                f"  {arm:8s} {label:14s} win={h['win']:.3f} draw={h['draw']:.3f} "
                f"loss={h['loss']:.3f}  delta={h['sd_mean']:+6.1f} +- {h['sd_std']:.1f}"
            )
            danger[label] = h["opp_win_map"]
        for label, plane in danger.items():
            print(f"           predicted opp danger ({label}): {top_squares(plane)}")


if __name__ == "__main__":
    main()
