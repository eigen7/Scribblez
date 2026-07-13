# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, single-threaded runs, NWL23 lexicon,
Release build). They characterize the `hastybot-endgame` agent's cost/strength
tradeoff as a function of its `--endgame-nodes` budget and are the basis for
the shipped default of 200.

## Methodology

- **Cost** (`--mode=games`): mean wall-time per game of
  endgame-vs-endgame self-play, as a ratio to the hasty-vs-hasty baseline
  (~5 ms/game, ~215 games/s single-threaded), same seeds across configs.
- **Strength** (`--mode=games` head-to-head): the endgame bot's mean final
  spread against a plain greedy HastyBot. Each seed is played twice with the
  seats mirrored and the pair averaged, which cancels per-seed tile-draw luck —
  the dominant variance source. Unpaired spread estimates swing by +/- 5+
  points below ~1000 games and are not usable.
- Aggregates below pool 6-8 independent seed shards per budget (900-4800 games
  per row); the quoted uncertainty is the standard error across shards.

## Cost and strength vs node budget

| `--endgame-nodes` | game-time cost | head-to-head vs greedy HastyBot |
|---|---|---|
| 100 | 1.15x | +0.15 +/- 0.02 pts/game |
| **200 (default)** | **2.00x** | **+0.40 +/- 0.03** |
| 400 | 4.5x | +0.85 +/- 0.07 |
| 1600 | 21x | +4.96 +/- 0.19 |
| 5000 | 73x | +5.48 +/- 0.22 |
| 20000 | 202x | +5.58 +/- 0.35 |

Strength saturates at roughly **+5.5 points/game — the full value of exact
endgame play over greedy static-equity play** in hasty-level self-play. Budget
1600 captures ~90% of the ceiling at a tenth of the cost of 20000.

Strength rises superlinearly through the low budgets because the budget gates
*engagement*, not search noise: a solve is declined up front when its root move
count exceeds the budget, and the agent falls back to HastyBot's static-equity
move unless the solver completed its first iteration. Raising the budget
admits richer (more consequential) endgames to a real search; it never makes
an admitted search noisier.

## Per-solve cost (`--mode=solves`, 80 captured bag-empty positions)

| budget | mean us | p50 us | p95 us | mean depth | % differ from hasty |
|---|---|---|---|---|---|
| 100 | 445 | 217 | 1552 | 1.6 | 38% |
| 200 | 1144 | 421 | 4293 | 2.6 | 38% |
| 400 | 3911 | 2591 | 17548 | 3.9 | 44% |
| 1600 | 37727 | 18456 | 140045 | 6.2 | 48% |
| 5000 | 121492 | 62604 | 412281 | 8.2 | 45% |
| 20000 | 405068 | 206320 | 1663822 | 9.7 | 49% |

Solves are movegen-bound (roughly 35-80k nodes/s; each node and each greedy
playout ply runs a full move generation), so per-solve cost tracks nodes spent
almost linearly once positions stop being declined.

## Reproducing

```
target/engine/endgame_bench --mode=games  --games=800 --seed=S --budgets=B --threads=1
target/engine/endgame_bench --mode=solves --games=30  --seed=42 --budgets=100,200,400,1600,5000,20000
```

For strength numbers, run several `--seed` shards per budget and pool them;
single-shard head-to-head spreads carry a few points of standard error even
with seat mirroring.
