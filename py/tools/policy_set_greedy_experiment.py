#!/usr/bin/env python3
"""Train one arm of the policy-set greedy form on a directory of labeled pool
chunks and report how well its gains fit their targets.

The arms:
  base        the greedy form as it trains (greedy.GreedyModel)
  mixed       the same model, half its training rows seeded from a draw
              toward high win probability instead of the top-equity move
              (greedy.sample_rows' drawn_seed_share)
  pair        a pairwise model: a head g(a, s) predicts candidate a's
              expected improvement over one member s from the cached
              candidate vectors, trained on every (candidate, member) target;
              a set's gain is the least g over its members
  pair_mixed  pair, with mixed seeds

The report scores every arm on deployment-style rows (the seed always the
top-equity move): mean target against mean prediction by equity band and set
size, on training and held-out pools -- predictions that track their targets
across set sizes, not flat ones, are what a set-aware gain head shows -- then
recall and regret against the static-equity baseline on the held-out pools.

    py/tools/policy_set_greedy_experiment.py pair \\
        --pools /workspace/mount/scratch-policy-set-pools-v5 [--epochs 12]

The pools are .slog + .forgone + .pool chunks, one directory each (as
policy_set_pool_tool writes them); the last --held-out chunks are held out.
"""

import argparse
import functools
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from cloud.sinks import LocalSink
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.generational.records import TrainRecorder
from scribblez.policy_set import dataset, forms, greedy, metrics, train_loop, trainer
from scribblez.policy_set.model import to_grid
from scribblez.workloads.policy_set import PolicySetParams
from torch import nn

ARMS = ("base", "mixed", "pair", "pair_mixed")
# The mixed arms' share of training rows whose seed is drawn.
DRAWN_SEED_SHARE = 0.5
# Short: the shakeout pools are a few thousand, not a tag's stream.
WARMUP_POOLS = 2000
# The table's rows: equity points below the pool's best; its columns: set
# sizes, the seed included.
EQUITY_BANDS = ((0, 1), (1, 3), (3, 10), (10, float("inf")))
SET_SIZE_BANDS = ((1, 1), (2, 4), (5, 9), (10, 15))
# Gaps below this many points count as near the top for the correlation line.
NEAR_TOP_POINTS = 3
TABLE_POOLS = 400
TABLE_ROWS_PER_POOL = 4
TABLE_SEED = 7


class PairModel(nn.Module):
    """g(a, s): candidate a's expected improvement over member s, in
    percentage points, from the two plain candidate vectors, their product
    and the two equity inputs. Nothing reruns per step: the vectors are
    computed once per position."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        c = encoder.board_pos_emb.shape[1]
        self.pair = nn.Sequential(
            nn.Linear(3 * c + 3, 2 * c),
            nn.ReLU(inplace=True),
            nn.Linear(2 * c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 1),
        )

    def cache(self, batch):
        """The candidate vectors (M, C) and equity inputs (M,)."""
        enc = self.encoder.encode(batch)
        h = self.encoder.candidate_vectors(enc.e, enc.board, enc.g, batch["move_pos_id"])
        return h, batch["move_equity"].to(h.dtype)

    def pair_gains(self, h, equity, a, s):
        """Global candidate indices a, s of one shape -> g(a, s) of that shape."""
        ea, es = equity[a].unsqueeze(-1), equity[s].unsqueeze(-1)
        x = torch.cat([h[a], h[s], h[a] * h[s], ea, es, ea - es], dim=-1)
        return self.pair(x).squeeze(-1)

    def gains(self, h, equity, pos_id, rows):
        """Each row's candidates' set gains (Mr,), the least over the row's
        members, and their pairwise gains (Mr, E), with the candidates' global
        indices and row ids."""
        move_idx, row_id = greedy.row_candidates(pos_id, rows.position)
        members = rows.chosen[row_id]
        g = self.pair_gains(h, equity, move_idx.unsqueeze(1).expand_as(members), members)
        least = g.float().masked_fill(~rows.mask[row_id], torch.inf).min(dim=1).values
        return least, g, move_idx, row_id


def pair_targets(batch, rows, move_idx, row_id):
    """Each row candidate's expected improvement over each member (Mr, E), in
    percentage points, and the pairs that count: members, not padding, and
    not the candidate itself."""
    values, rollouts = batch["rollout_values"].float(), batch["rollouts"]
    members, in_set = rows.chosen[row_id], rows.mask[row_id]
    columns = [
        greedy.expected_improvement(*greedy.paired_difference(values, rollouts, move_idx, s))
        for s in members.unbind(dim=1)
    ]
    valid = in_set & (members != move_idx.unsqueeze(1))
    return torch.stack(columns, dim=1) * greedy.GAIN_SCALE, valid


def training_rows(batch, params, rng, drawn_seed_share):
    position, chosen = greedy.sample_rows(
        batch["win_prob"].numpy(force=True), batch["move_equity"].numpy(force=True),
        batch["move_pos_id"].numpy(force=True), params.greedy_rows, rng, drawn_seed_share,
    )  # fmt: skip
    return greedy.make_rows(position, chosen).to(batch["move_pos_id"].device)


def greedy_loss(model, batch, params, rng, drawn_seed_share):
    return {
        "gain": greedy.gain_loss(model, batch, training_rows(batch, params, rng, drawn_seed_share))
    }


def pair_loss(model, batch, params, rng, drawn_seed_share):
    """Squared error on every valid pair's gain."""
    rows = training_rows(batch, params, rng, drawn_seed_share)
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        h, equity = model.cache(batch)
        _, g, move_idx, row_id = model.gains(h, equity, batch["move_pos_id"], rows)
    target, valid = pair_targets(batch, rows, move_idx, row_id)
    return {"gain": F.mse_loss(g.float()[valid], target[valid])}


@torch.no_grad()
def pair_picks(model, batch, n):
    """greedy.greedy_picks for the pairwise model: each step lowers every
    candidate's gain to its g over the newest member."""
    pos_id = batch["move_pos_id"]
    device = pos_id.device
    num_positions = batch["input_spatial"].shape[0]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        h, equity = model.cache(batch)
    equity_grid, _ = to_grid(batch["move_equity"].float(), pos_id, num_positions, -torch.inf)
    sizes = torch.bincount(pos_id, minlength=num_positions)
    starts = torch.cumsum(sizes, 0) - sizes
    newest = starts + equity_grid.argmax(dim=1)
    chosen = [[int(s)] for s in newest]
    every = torch.arange(len(pos_id), device=device)
    gain = torch.full((len(pos_id),), torch.inf, device=device)
    for _ in range(min(n, int(sizes.max())) - 1):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gain = torch.minimum(gain, model.pair_gains(h, equity, every, newest[pos_id]).float())
        open_gain = gain.clone()
        open_gain[torch.tensor([i for c in chosen for i in c], device=device)] = -torch.inf
        grid, _ = to_grid(open_gain, pos_id, num_positions, -torch.inf)
        newest = starts + grid.argmax(dim=1)
        for p in range(num_positions):
            if len(chosen[p]) < int(sizes[p]):
                chosen[p].append(int(newest[p]))
    return [[i - int(starts[p]) for i in c] for p, c in enumerate(chosen)]


def arm_form(arm: str) -> forms.Form:
    share = DRAWN_SEED_SHARE if arm.endswith("mixed") else 0.0
    if arm.startswith("pair"):
        return forms.Form(
            lambda encoder, params: PairModel(encoder),
            functools.partial(pair_loss, drawn_seed_share=share),
            pair_picks,
            lambda model, batch: {},
        )
    return replace(
        forms.FORMS[forms.FORM_GREEDY], loss=functools.partial(greedy_loss, drawn_seed_share=share)
    )


def train(arm, items, params, epochs, device):
    """The arm's model, trained `epochs` passes over `items`, in eval mode."""
    forms.FORMS[arm] = arm_form(arm)
    models = nn.ModuleDict({arm: forms.FORMS[arm].build(trainer.build_encoder(params), params)})
    models = models.to(device)
    optimizer = build_optimizer(models, params)
    optim_arm = build_optim_arm(
        TrainRecorder(LocalSink(Path(tempfile.mkdtemp()))), params, optimizer, 0
    )
    batcher = dataset.PoolBatcher()
    rows = 0
    for epoch in range(epochs):
        t0 = time.time()
        optim_arm.train_mode()
        result = train_loop.run_epoch(
            models, optimizer, batcher, items, device, params,
            seed=epoch, lr_fn=optim_arm.lr_fn, rows_trained=rows,
        )  # fmt: skip
        rows = result.rows_trained
        print(
            f"epoch {epoch} loss {result.loss:.4f} grad norm {result.grad_norms[arm]:.3f} "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    recalibration = trainer.recalibration_batches(batcher, items, params, epochs, device)
    optim_arm.eval_mode(models, recalibration, trainer._trunk_forward)
    return models[arm].eval()


@torch.no_grad()
def set_gains(model, batch, rows):
    """Either kind of model's set gains on `rows`: (gains, candidates, row ids)."""
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        if isinstance(model, PairModel):
            h, equity = model.cache(batch)
            g, _, move_idx, row_id = model.gains(h, equity, batch["move_pos_id"], rows)
        else:
            enc, h = model.cache(batch)
            g, move_idx, row_id = model.gains(enc, h, batch["move_pos_id"], rows)
    return g.float(), move_idx, row_id


@torch.no_grad()
def fit_samples(model, items, device) -> dict[str, np.ndarray]:
    """Per row candidate on deployment-style rows: its equity gap (points),
    target and prediction (pp), its row's set size, and whether it is a
    member."""
    rng = np.random.default_rng(TABLE_SEED)
    batcher = dataset.PoolBatcher()
    out = {k: [] for k in ("gap", "target", "pred", "size", "member")}
    for start in range(0, len(items), 8):
        batch = train_loop.to_device(batcher.build(items[start : start + 8]), device)
        position, chosen = greedy.sample_rows(
            batch["win_prob"].numpy(force=True), batch["move_equity"].numpy(force=True),
            batch["move_pos_id"].numpy(force=True), TABLE_ROWS_PER_POOL, rng,
        )  # fmt: skip
        rows = greedy.make_rows(position, chosen).to(device)
        pred, move_idx, row_id = set_gains(model, batch, rows)
        target = greedy.gain_targets(batch, rows, move_idx, row_id) * greedy.GAIN_SCALE
        member = ((rows.chosen[row_id] == move_idx.unsqueeze(1)) & rows.mask[row_id]).any(1)
        # The equity input is stored /10: back to points below the best.
        out["gap"].append(-batch["move_equity"].float()[move_idx] * 10)
        out["target"].append(target)
        out["pred"].append(pred)
        out["size"].append(rows.mask.sum(1)[row_id])
        out["member"].append(member)
    return {k: torch.cat(v).numpy(force=True) for k, v in out.items()}


def print_fit_table(name, d):
    outside = ~d["member"]
    print(f"\n{name}: mean target / mean prediction (pp) outside the set; n in brackets")
    print(f"{'gap \\ size':>12}" + "".join(f"{f'{lo}-{hi}':>22}" for lo, hi in SET_SIZE_BANDS))
    for g_lo, g_hi in EQUITY_BANDS:
        line = f"{f'[{g_lo},{g_hi})':>12}"
        for s_lo, s_hi in SET_SIZE_BANDS:
            band = (d["gap"] >= g_lo) & (d["gap"] < g_hi)
            k = outside & band & (d["size"] >= s_lo) & (d["size"] <= s_hi)
            cell = f"{d['target'][k].mean():8.2f} /{d['pred'][k].mean():6.2f} ({k.sum():5d})"
            line += cell if k.any() else f"{'-':>22}"
        print(line)
    near = outside & (d["gap"] < NEAR_TOP_POINTS)
    near_corr = np.corrcoef(d["target"][near], d["pred"][near])[0, 1]
    all_corr = np.corrcoef(d["target"][outside], d["pred"][outside])[0, 1]
    print(
        f"  corr(target, pred): near the top {near_corr:.3f}, all {all_corr:.3f}; "
        f"MSE {np.mean((d['target'][outside] - d['pred'][outside]) ** 2):.4f}"
    )


@torch.no_grad()
def print_set_metrics(arm, model, held, device):
    batcher = dataset.PoolBatcher()
    picks = []
    for start in range(0, len(held), 8):
        batch = train_loop.to_device(batcher.build(held[start : start + 8]), device)
        picks += forms.FORMS[arm].sets(model, batch, max(metrics.SET_SIZES))
    by_pool = {id(it.pool): p for it, p in zip(held, picks, strict=True)}
    pools = [it.pool for it in held]
    chooser = functools.partial(train_loop.precomputed_pick, by_pool)
    ours = metrics.set_metrics(pools, chooser, "")
    baseline = metrics.set_metrics(pools, metrics.equity_top, "")
    print(f"\nheld out ({len(held)} pools)  N: recall model / equity   regret (pp) model / equity")
    for n in metrics.SET_SIZES:
        recall, regret = f"eval_recall_{n}", f"eval_regret_{n}"
        print(
            f"  {n:2d}: {ours[recall]:.3f} / {baseline[recall]:.3f}"
            f"    {100 * ours[regret]:.3f} / {100 * baseline[regret]:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--pools", type=Path, required=True, help="a directory of pool chunks")
    parser.add_argument("--held-out", type=int, default=2, help="chunks held out, the last ones")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    chunks = sorted(p for p in args.pools.iterdir() if p.is_dir() and any(p.glob("*.pool")))
    items = dataset.read_items(chunks[: -args.held_out])
    held = dataset.read_items(chunks[-args.held_out :])
    params = replace(PolicySetParams(), forms=forms.FORM_GREEDY, lr_warmup_rows=WARMUP_POOLS)
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(f"arm {args.arm}: {len(items)} training pools, {len(held)} held out", flush=True)
    model = train(args.arm, items, params, args.epochs, device)
    print_fit_table(
        f"TRAINING pools (first {TABLE_POOLS})", fit_samples(model, items[:TABLE_POOLS], device)
    )
    print_fit_table("HELD-OUT pools", fit_samples(model, held, device))
    print_set_metrics(args.arm, model, held, device)


if __name__ == "__main__":
    main()
