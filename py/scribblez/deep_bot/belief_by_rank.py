"""The leave inference model's score by the static-equity rank of the move
played (docs/plans/deep_bot_generations.md, "Measurements"): the belief
measurement the generation loop exists to move.

After a move HastyBot would make, the leave says little that static equity
does not; after one it ranks low, the mover saw something, and what she kept
is the tell. A model trained on the agent's games should gain most there. The
overall score cannot show it, since most moves played are equity's first.

A turn's pool gives the played move's rank; the row scored is the opponent's
at the next turn, whose target is the leave that move kept.
"""

from pathlib import Path

import numpy as np
import torch

from scribblez import ffi
from scribblez.dataset import row_layout, slice_row_batch
from scribblez.leave_inference import loss
from scribblez.leave_inference.train_loop import batch_nll, to_device
from scribblez.leave_inference.unit import task
from scribblez.policy_set import pools
from scribblez.sim_evidence import slog_meta

# The rank bands scored, by name: (lowest, highest) rank, 0 equity's first.
BANDS = {"0": (0, 0), "1_9": (1, 9), "10_up": (10, np.iinfo(np.int32).max)}


def has_leave_row(meta, turn: int) -> bool:
    """Whether `turn` of the game `meta` describes has a leave inference
    row: the rows cover the game's training-eligible turns, which stop short
    of its end."""
    return int(meta["eligible_begin"]) <= turn < int(meta["eligible_end"])


def played_ranks(pool_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pool of `pool_path` whose game's move is one of its ranked
    candidates and whose next turn has a leave inference row: (games, turns,
    ranks), the turn the move was played at and its static-equity rank."""
    slog = slog_meta.read_slog_bytes(pool_path.with_suffix(".slog"))
    metas = slog_meta.game_metas(slog)
    out = []
    for pool in pools.read_pools(pool_path)[1]:
        meta = metas[pool.game]
        if not has_leave_row(meta, pool.turn + 1):
            continue
        played = np.flatnonzero(pool.moves == slog_meta.move_at(slog, meta, pool.turn))
        if len(played) == 1 and pool.equity_rank[played[0]] >= 0:
            out.append((pool.game, pool.turn, int(pool.equity_rank[played[0]])))
    games, turns, ranks = np.array(out, dtype=np.int64).reshape(-1, 3).T
    return games, turns, ranks


@torch.no_grad()
def nll_vs_uniform(unit, slog: Path, games: np.ndarray, turns: np.ndarray):
    """The model's NLL less the uniform draw's for the leave inference rows
    at (games[j], turns[j]) of `slog`, from each observer's own view: (rows,)
    with NaN where the row has no leave to infer."""
    name = task(unit.params)
    rows = ffi.decode_rows(slog, games, turns, post_move=False, task=name)
    layout = row_layout(ffi.task_input_shapes(name), ffi.task_target_shapes(name))
    batch = to_device(slice_row_batch(rows, *layout), unit.device)
    gap = np.full(len(rows), np.nan)
    if batch.valid.any():
        nll = batch_nll(unit.model, unit.table, batch).nll
        uniform = loss.uniform_nll(batch.leave_counts, batch.length, batch.pool)
        gap[batch.valid.cpu().numpy()] = (nll - uniform).float().cpu().numpy()
    return gap


def evaluate(unit, gen_dir: Path) -> dict[str, float]:
    """`unit`'s (a LeaveInferenceUnit's) score on the games of `gen_dir` by
    the rank of the move played, keyed "eval_nll_vs_uniform_rank_<band>" for
    each band that has rows, with each band's share of the rows scored,
    "eval_played_rank_<band>_share"."""
    unit.model.eval()
    gaps, ranks = [np.zeros(0)], [np.zeros(0, dtype=np.int64)]
    for pool_path in sorted(gen_dir.glob(f"*{pools.SUFFIX}")):
        games, turns, rank = played_ranks(pool_path)
        if len(games) == 0:
            continue
        # The opponent's next turn is where the kept leave is the target.
        gaps.append(nll_vs_uniform(unit, pool_path.with_suffix(".slog"), games, turns + 1))
        ranks.append(rank)
    gap, rank = np.concatenate(gaps), np.concatenate(ranks)
    scored = ~np.isnan(gap)
    out = {}
    for band, (lo, hi) in BANDS.items():
        sel = scored & (rank >= lo) & (rank <= hi)
        if sel.any():
            out[f"eval_nll_vs_uniform_rank_{band}"] = float(gap[sel].mean())
            out[f"eval_played_rank_{band}_share"] = float(sel.sum() / scored.sum())
    return out
