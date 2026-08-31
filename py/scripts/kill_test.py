#!/usr/bin/env python3
"""The 4-armed sim-evidence kill-test (docs/sim_residual_feedback.md, step 3).

Tests the load-bearing hypothesis of the sim-evidence loop in isolation: does
conditioning the position evaluation model on Monte-Carlo sim evidence improve
its outcome prediction? One invocation runs the whole experiment against the data
accumulated by scripts/generate_kill_test_data.py under the same tag:

  1. Cache build: every complete .slog/.sobs pair in the tag's pair store
     (<mount>/tags/kill_test/<tag>/data/slogs) is decoded (each evidence
     position's post-move training row, addressed by identity) and its evidence
     set encoded, into one .npz shard per file under the tag's cache/ dir.
     Shards are cached across invocations; only new files are processed.
     Files alternate into train/holdout round-robin (--holdout-every), so the
     split is by game and cannot leak.

  2. Four training arms, identical seed and architecture (the fusion stage is
     zero-initialized and parameter counts match across arms; only the
     evidence input differs). Every evidence arm is leave-one-out by default:
     the played move's own sim (candidate 0) is masked, because at deployment
     the model only re-scores unsimmed moves -- simmed ones are ranked by
     their sims directly -- and the own-sim token is a near-copy of the
     training target.
       none      evidence zeroed -- the baseline
       shuffled  real (leave-one-out) evidence permuted across positions -- a
                 falsification control; any gain over `none` is what the
                 model extracts from evidence marginals rather than
                 position-matched evidence
       scalar    scalar sim summaries only (the cheap rung of the ladder)
       full      spatial planes + scalar summaries (the real thing)
     A fifth arm, `ownsim` (--arms ownsim), is the opt-in non-LOO variant:
     full evidence including the played move's own sim. `ownsim` vs `full`
     prices the own-sim shortcut; it is not a deployment-relevant capability.

  Passing --open-leaves selects the open-leaves information condition end to
  end: the cache decodes rows with the opponent-leave input block, and the
  .sobs sidecars are required to carry the open-leaves flag (i.e. generated
  by generate_kill_test_data.py --open-leaves, whose sims start the opponent
  from the leave their last move retained, with replenishments sampled).
  Open-leaves and hidden artifacts must live under different tags; the cache
  records its mode and refuses a mismatch.

The decision metric is best held-out WLD cross-entropy; a comparison table
prints at the end and per-arm histories land in the tag's cache/results/ dir.
See the doc's kill-test section for the decision rubric.

After the arms, a paired per-position analysis runs automatically (and can be
rerun without retraining via --analyze): per-position CE deltas with sign
tests, sliced by whether the sim's opponent-rack sampling is exact at the
position (opponent just bingoed / hasn't acted) and by game phase, plus an
evidence-only logistic yardstick measuring how predictive the raw sim scalars
are without any board input. Caches built before these analyses existed are
upgraded in place (meta columns are backfilled from the .slog/.sobs headers;
no row re-decode).

Usage:
    ./py/scripts/kill_test.py -t apple
    ./py/scripts/kill_test.py -t apple --analyze   # re-print analysis only
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scribblez.dataset import row_layout
from scribblez.ffi import decode_rows, get_input_shapes, set_opp_leave_input
from scribblez.sim_evidence.model import EvidencePositionEvalModel
from scribblez.sim_evidence.slog_meta import position_meta
from scribblez.sim_evidence.sobs import (
    NUM_EVIDENCE_SCALARS,
    SOBS_FLAG_OPEN_LEAVES,
    SOBS_FLAG_RETIRED_OPEN_RACK,
    evidence_features,
    read_sobs,
    read_sobs_flags,
)
from scribblez.workloads import kill_test as kill_test_workload

ARMS = ("none", "shuffled", "scalar", "full")


# ---------------------------------------------------------------------------
# cache build
# ---------------------------------------------------------------------------


def build_shard(slog: Path, sobs: Path, max_k: int) -> dict[str, np.ndarray]:
    """Decode one file's evidence positions into a shard's arrays."""
    positions = read_sobs(sobs)
    games = np.array([p.game_index for p in positions], dtype=np.int64)
    turns = np.array([p.turn_index for p in positions], dtype=np.int64)
    rows = decode_rows(slog, games, turns, post_move=True)

    input_shapes, targets = row_layout()
    spatial_shape = input_shapes[0].dims
    scalar_width = input_shapes[1].dims[0]
    spatial_floats = int(np.prod(spatial_shape))

    n = len(positions)
    ev_planes = np.zeros((n, max_k, 5, 15, 15), dtype=np.float16)
    ev_scalars = np.zeros((n, max_k, NUM_EVIDENCE_SCALARS), dtype=np.float32)
    ev_mask = np.zeros((n, max_k), dtype=bool)
    for i, pos in enumerate(positions):
        ev_planes[i], ev_scalars[i], ev_mask[i] = evidence_features(pos, max_k)

    shard = {
        "input_spatial": rows[:, :spatial_floats].reshape(n, *spatial_shape).astype(np.float16),
        "input_scalar": rows[:, spatial_floats : spatial_floats + scalar_width],
        "ev_planes": ev_planes,
        "ev_scalars": ev_scalars,
        "ev_mask": ev_mask,
        **position_meta(slog, positions),
    }
    for name, start, end, dims in targets:
        shard[name] = rows[:, start:end].reshape(n, *dims)
    return shard


def build_cache(slog_dir: Path, cache: Path, holdout_every: int, max_k: int, open_leaves: bool):
    (cache / "train").mkdir(parents=True, exist_ok=True)
    (cache / "holdout").mkdir(parents=True, exist_ok=True)

    # The cache's information condition is fixed at first build (its rows'
    # input width depends on it); later builds and training must match.
    mode_path = cache / "mode.json"
    mode = {"open_leaves": open_leaves}
    if mode_path.exists() and json.loads(mode_path.read_text()) != mode:
        raise SystemExit(f"{mode_path} disagrees with --open-leaves; use a separate tag per mode")
    mode_path.write_text(json.dumps(mode) + "\n")

    slogs = sorted(slog_dir.glob("*.slog"))
    pairs = [(s, s.with_suffix(".sobs")) for s in slogs if s.with_suffix(".sobs").exists()]
    if not pairs:
        raise SystemExit(
            f"no .slog/.sobs pairs under {slog_dir} (run generate_kill_test_data.py first)"
        )
    for _, sobs in pairs:
        flags = read_sobs_flags(sobs)
        if flags & SOBS_FLAG_RETIRED_OPEN_RACK:
            raise SystemExit(
                f"{sobs} was generated by the retired full-open-rack mode; regenerate it "
                "(the open-leaves condition reveals only the opponent's retained leave)"
            )
        sobs_open = bool(flags & SOBS_FLAG_OPEN_LEAVES)
        if sobs_open != open_leaves:
            raise SystemExit(
                f"{sobs} was generated with open_leaves={sobs_open} but this run has "
                f"open_leaves={open_leaves}; regenerate under a dedicated tag"
            )
    print(f"{len(pairs)} .slog/.sobs pairs; holdout = every {holdout_every}th file")

    added = upgraded = 0
    for i, (slog, sobs) in enumerate(pairs):
        split = "holdout" if i % holdout_every == 0 else "train"
        out = cache / split / f"{slog.stem}.npz"
        if out.exists():
            upgraded += upgrade_shard_meta(out, slog, sobs)
            continue
        shard = build_shard(slog, sobs, max_k)
        np.savez_compressed(out, **shard)
        added += len(shard["ev_mask"])
        print(f"  [{split}] {out.name}: {len(shard['ev_mask'])} positions")
    print(f"cache up to date ({added} positions added, {upgraded} shards meta-upgraded)")


def upgrade_shard_meta(shard_path: Path, slog: Path, sobs: Path) -> int:
    """Backfill the analysis meta columns into a shard written before they
    existed. Reads only .sobs positions and .slog headers (no row decode), so
    upgrading a cache is cheap. Returns 1 if the shard was rewritten."""
    with np.load(shard_path) as z:
        if "meta_opp_unbiased" in z.files:
            return 0
        columns = {key: z[key] for key in z.files}
    columns.update(position_meta(slog, read_sobs(sobs)))
    np.savez_compressed(shard_path, **columns)
    return 1


# ---------------------------------------------------------------------------
# training arms
# ---------------------------------------------------------------------------


def load_split(cache: Path, split: str) -> dict[str, torch.Tensor]:
    """Concatenate a split's shards into one in-memory tensor dict."""
    shards = sorted((cache / split).glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {cache / split}")
    columns: dict[str, list[np.ndarray]] = {}
    for path in shards:
        with np.load(path) as z:
            for key in z.files:
                columns.setdefault(key, []).append(z[key])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in columns.items()}


def apply_evidence_mode(data: dict[str, torch.Tensor], mode: str, seed: int):
    """Mutate a split's evidence tensors in place per the arm being trained.

    Every evidence arm except `ownsim` is leave-one-out: candidate 0 -- the
    equity argmax, which on HastyBot self-play is the played move whose
    post-move state the row encodes -- is masked out. What remains is the
    deployment-shaped query: evaluate a move given the OTHER candidates' sims
    (at deployment the model only ever re-scores unsimmed moves; simmed ones
    are ranked by their own sims directly). `ownsim` opts back into the full
    evidence set; its gain over `full` prices the own-sim shortcut, whose
    token is a near-copy of the training target."""
    if mode == "none":
        data["ev_planes"].zero_()
        data["ev_scalars"].zero_()
        data["ev_mask"].zero_()
        return
    if mode not in ("full", "scalar", "shuffled", "ownsim"):
        raise ValueError(f"unknown evidence mode {mode}")
    if mode != "ownsim":
        data["ev_planes"][:, 0] = 0
        data["ev_scalars"][:, 0] = 0
        data["ev_mask"][:, 0] = False
    if mode == "scalar":
        data["ev_planes"].zero_()
    elif mode == "shuffled":
        perm = torch.randperm(len(data["ev_mask"]), generator=torch.Generator().manual_seed(seed))
        for key in ("ev_planes", "ev_scalars", "ev_mask"):
            data[key] = data[key][perm]


def batch_slices(n: int, batch_size: int, generator: torch.Generator | None):
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for lo in range(0, n, batch_size):
        yield order[lo : lo + batch_size]


def to_device(data: dict[str, torch.Tensor], idx: torch.Tensor, device) -> dict[str, torch.Tensor]:
    out = {}
    for key, tensor in data.items():
        t = tensor[idx].to(device)
        if t.dtype == torch.float16:
            t = t.float()
        out[key] = t
    return out


def forward(model, batch):
    return model(
        batch["input_spatial"],
        batch["input_scalar"],
        batch["ev_planes"],
        batch["ev_scalars"],
        batch["ev_mask"],
    )


@torch.no_grad()
def evaluate(model, data, device, batch_size: int) -> tuple[dict[str, float], np.ndarray]:
    """Held-out metrics plus the per-position WLD cross-entropy vector (row
    order = the split's shard order, which the paired analysis relies on).
    wld_ce is the kill-test's decision metric."""
    model.eval()
    n = len(data["wld"])
    per_row = np.zeros(n, dtype=np.float32)
    brier_sum = correct = sd_ae_sum = 0.0
    for idx in batch_slices(n, batch_size, generator=None):
        batch = to_device(data, idx, device)
        out = forward(model, batch)
        target_idx = batch["wld"].argmax(dim=1)
        probs = out["wld"].softmax(dim=1)
        per_row[idx.numpy()] = (
            F.cross_entropy(out["wld"], target_idx, reduction="none").cpu().numpy()
        )
        brier_sum += ((probs - batch["wld"]) ** 2).sum().item()
        correct += (out["wld"].argmax(1) == target_idx).sum().item()
        sd_ae_sum += (out["score_diff"][:, 0] - batch["score_diff"].squeeze(1)).abs().sum().item()
    model.train()
    metrics = {
        "wld_ce": float(per_row.mean()),
        "brier": brier_sum / n,
        "wld_acc": correct / n,
        "sd_mae": sd_ae_sum / n,
    }
    return metrics, per_row


def train_arm(arm: str, cache: Path, args, device) -> dict:
    """Train one arm from scratch and return its result record."""
    torch.manual_seed(args.seed)
    train = load_split(cache, "train")
    holdout = load_split(cache, "holdout")
    apply_evidence_mode(train, arm, seed=args.seed)
    apply_evidence_mode(holdout, arm, seed=args.seed + 1)

    input_shapes = {s.name: s.dims for s in get_input_shapes()}
    model = EvidencePositionEvalModel(
        spatial_planes=input_shapes["input_spatial"][0],
        scalar_size=input_shapes["input_scalar"][0],
        trunk_channels=args.trunk_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n=== arm={arm} train={len(train['wld'])} holdout={len(holdout['wld'])} "
        f"params={n_params:,} device={device} ==="
    )

    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best_per_row: np.ndarray | None = None
    best_state: dict | None = None
    best_ce = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0
        for idx in batch_slices(len(train["wld"]), args.batch_size, generator):
            batch = to_device(train, idx, device)
            out = forward(model, batch)
            losses = model.compute_loss(
                out,
                {k: batch[k] for k in model.target_keys()},
                lambda_sd=args.lambda_sd,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            loss_sum += losses["total"].item()
            n_batches += 1

        metrics, per_row = evaluate(model, holdout, device, args.batch_size)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss_sum / max(n_batches, 1)
        history.append(metrics)
        if metrics["wld_ce"] < best_ce:
            best_ce = metrics["wld_ce"]
            best_per_row = per_row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:3d}  train_loss={metrics['train_loss']:.4f}  "
            f"holdout: wld_ce={metrics['wld_ce']:.4f} brier={metrics['brier']:.4f} "
            f"acc={metrics['wld_acc']:.4f} sd_mae={metrics['sd_mae']:.1f}  "
            f"({time.time() - t0:.0f}s)"
        )
        best_epoch = min(history, key=lambda m: m["wld_ce"])["epoch"]
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no holdout improvement in {args.patience} epochs")
            break

    best = min(history, key=lambda m: m["wld_ce"])
    return {
        "arm": arm,
        "history": history,
        "best": best,
        "per_row_ce": best_per_row,
        "state_dict": best_state,
    }


def paired_stats(delta: np.ndarray) -> str:
    """Mean +/- SE of per-position CE deltas, plus the sign-test win rate and
    its two-sided normal-approximation p-value."""
    n = len(delta)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n))
    wins = int((delta < 0).sum())  # negative delta = first arm better
    z = (wins - n / 2) / np.sqrt(n / 4)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return f"d={mean:+.4f} +/- {se:.4f}   win%={100 * wins / n:5.1f}   sign-p={p:.2g}"


def load_per_row(results_dir: Path, arm: str) -> np.ndarray:
    path = results_dir / f"{arm}_holdout_ce.npy"
    if not path.exists():
        raise SystemExit(
            f"{path} missing -- per-position losses are saved during training, so rerun "
            "the arms once (a run predating this analysis has none to analyze)"
        )
    return np.load(path)


def evidence_yardstick(train, holdout, device) -> dict[str, float]:
    """Holdout WLD cross-entropy of logistic regressions over the evidence
    scalars alone (no board, no trunk): how predictive is the raw sim output?
    `played move` uses only candidate 0 (the move actually played, whose sim
    estimates the target directly); `all candidates` uses every token."""

    def flat(data, k):
        m = data["ev_mask"][:, :k].float().unsqueeze(-1)
        return (data["ev_scalars"][:, :k] * m).flatten(1).to(device)

    y_tr = train["wld"].argmax(1).to(device)
    y_ho = holdout["wld"].argmax(1).to(device)
    out = {}
    for label, k in (("played move", 1), ("all candidates", train["ev_scalars"].shape[1])):
        x_tr, x_ho = flat(train, k), flat(holdout, k)
        model = torch.nn.Linear(x_tr.shape[1], 3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            F.cross_entropy(model(x_tr), y_tr).backward()
            opt.step()
        with torch.no_grad():
            out[label] = float(F.cross_entropy(model(x_ho), y_ho).item())
    return out


def run_analysis(cache: Path, device, suffix: str = ""):
    """Paired per-position comparison of the trained arms over the holdout
    split, sliced by the rack-inference and game-phase metadata, plus the
    evidence-only yardstick. Uses the per-position CE vectors saved at each
    arm's best epoch; row order is the holdout shard order in every artifact,
    which is what makes the pairing valid. Analyzes whichever arms (of the
    standard four plus `loo`) have saved artifacts under the given suffix, and
    drops arms whose holdout size differs from the current cache (arms trained
    before more data was generated cannot be paired)."""
    results_dir = cache / "results"
    holdout = load_split(cache, "holdout")
    if "meta_opp_unbiased" not in holdout:
        raise SystemExit("holdout shards lack meta columns -- rerun so build_cache upgrades them")
    n = len(holdout["wld"])

    ce = {}
    for arm in (*ARMS, "ownsim"):
        path = results_dir / f"{arm}{suffix}_holdout_ce.npy"
        if not path.exists():
            continue
        per_row = np.load(path)
        if len(per_row) != n:
            print(f"  (skipping arm `{arm}`: holdout size {len(per_row)} != cache {n})")
            continue
        ce[arm] = per_row
    if len(ce) < 2:
        print("analysis needs at least two arms with current-cache artifacts; skipping")
        return

    print("\n=== paired per-position analysis (negative d = first arm better) ===")
    pairs = [
        ("full", "none"),
        ("scalar", "none"),
        ("ownsim", "none"),
        ("ownsim", "full"),
        ("full", "shuffled"),
        ("full", "scalar"),
        ("shuffled", "none"),
    ]
    for a, b in pairs:
        if a in ce and b in ce:
            print(f"  {a:9s} vs {b:9s}  {paired_stats(ce[a] - ce[b])}")

    unbiased = holdout["meta_opp_unbiased"].numpy().astype(bool)
    remaining = holdout["meta_remaining"].numpy()
    terciles = np.quantile(remaining, [1 / 3, 2 / 3])
    slices = [
        (f"opp rack unbiased (n={unbiased.sum()})", unbiased),
        (f"opp rack biased   (n={(~unbiased).sum()})", ~unbiased),
        (f"late game, <= {terciles[0]:.0f} moves left", remaining <= terciles[0]),
        ("mid game", (remaining > terciles[0]) & (remaining <= terciles[1])),
        (f"early game, > {terciles[1]:.0f} moves left", remaining > terciles[1]),
    ]
    for a, b in (("full", "none"), ("scalar", "none"), ("ownsim", "none")):
        if a not in ce or b not in ce:
            continue
        print(f"\n=== {a} vs {b} by slice ===")
        delta = ce[a] - ce[b]
        for label, mask in slices:
            print(f"  {label:34s} {paired_stats(delta[mask])}")

    print("\n=== evidence-only yardstick (logistic on sim scalars; no board input) ===")
    train = load_split(cache, "train")
    for label, value in evidence_yardstick(train, holdout, device).items():
        print(f"  {label:16s} holdout wld_ce={value:.4f}")
    print(
        "  (compare against the arms above: if these sit well above the trunk arms,\n"
        "   the trunk already knows most of what a noisy S-rollout sim knows about\n"
        "   the root value, and evidence gains are bounded by the sim's own quality)"
    )


def print_summary(results: list[dict]):
    by_arm = {r["arm"]: r["best"] for r in results}
    base_ce = by_arm["none"]["wld_ce"]
    print("\n=== kill-test summary (best held-out epoch per arm) ===")
    print(f"{'arm':10s} {'wld_ce':>8s} {'d_vs_none':>10s} {'brier':>8s} {'acc':>7s} {'epoch':>6s}")
    for arm in ARMS:
        b = by_arm[arm]
        print(
            f"{arm:10s} {b['wld_ce']:8.4f} {b['wld_ce'] - base_ce:+10.4f} "
            f"{b['brier']:8.4f} {b['wld_acc']:7.4f} {b['epoch']:6d}"
        )
    print(
        "\nDecision rubric (docs/sim_residual_feedback.md): the hypothesis survives if\n"
        "`full` beats `none` by a margin that (a) dwarfs seed noise and (b) holds up\n"
        "against `shuffled` (which bounds what evidence marginals alone provide).\n"
        "`scalar` vs `full` locates how much of the win needs the spatial planes."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-t", "--tag", required=True, help="tag used with generate_kill_test_data.py")
    p.add_argument("--holdout-every", type=int, default=10, help="every Nth file is holdout")
    p.add_argument("--max-k", type=int, default=10, help="evidence candidates kept per position")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--lambda-sd",
        type=float,
        default=0.004,
        help="Score-diff loss weight (the production trainer's default), so the "
        "WLD head -- the decision metric -- dominates the objective.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--trunk-channels", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--patience",
        type=int,
        default=4,
        help="stop an arm after this many epochs without holdout improvement",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="skip training; rerun the paired/sliced analysis from saved artifacts",
    )
    p.add_argument(
        "--out-suffix",
        default="",
        help="appended to result filenames (<arm><suffix>.json etc.), so variant runs "
        "(e.g. a different --lambda-sd) do not overwrite the main arms",
    )
    p.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated subset of arms to run (default: the four standard arms, "
        "whose evidence is leave-one-out; `ownsim` is additionally available -- full "
        "evidence including the played move's own sim, pricing that shortcut)",
    )
    p.add_argument(
        "--open-leaves",
        action="store_true",
        help="the open-leaves information condition: decode rows with the opponent-leave "
        "input block and require open-leaves .sobs sidecars (a dedicated tag generated "
        "with generate_kill_test_data.py --open-leaves)",
    )
    args = p.parse_args()

    root = kill_test_workload.SPEC.data_dir(args.tag)
    cache = root / "cache"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = cache / "results"

    # The session's input layout is fixed at creation; choose the arm before
    # any FFI call (shape queries included).
    set_opp_leave_input(args.open_leaves)

    if args.analyze:
        run_analysis(cache, device, suffix=args.out_suffix)
        return

    build_cache(
        kill_test_workload.slog_dir(args.tag), cache, args.holdout_every, args.max_k,
        args.open_leaves,
    )  # fmt: skip
    results_dir.mkdir(exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = []
    for arm in arms:
        record = train_arm(arm, cache, args, device)
        name = f"{arm}{args.out_suffix}"
        np.save(results_dir / f"{name}_holdout_ce.npy", record.pop("per_row_ce"))
        torch.save(record.pop("state_dict"), results_dir / f"{name}_model.pt")
        arg_record = {k: v for k, v in vars(args).items() if k != "arms"}
        (results_dir / f"{name}.json").write_text(
            json.dumps({"args": arg_record, **record}, indent=2) + "\n"
        )
        results.append(record)

    if set(arms) >= set(ARMS) and not args.out_suffix:
        print_summary(results)
    run_analysis(cache, device, suffix=args.out_suffix)
    print(f"per-arm histories: {results_dir}")


if __name__ == "__main__":
    main()
