# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, single-threaded runs, NWL23 lexicon,
Release build). They characterize the `hastybot-endgame` agent's cost/strength
tradeoff as a function of its `--endgame-nodes` budget and are the basis for
the shipped default of 200.

**Hardware**: Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside
the project's Docker container, Release build.

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

## First-win (WLD) mode

`--endgame-wld` (agent: `EndgameHastyBotAgent::Params.endgame_wld`) pins the
solver's root alpha-beta window to `(EndgameSolver::kFirstWinAlpha,
EndgameSolver::kFirstWinBeta)` = `(-1, +1)` instead of the full
`(-inf, +inf)` spread window. The search then only has to resolve which side
of an even final spread the position falls on -- win, draw, or loss -- rather
than the exact point margin, so proofs land with far less search effort per
ply of true difficulty. The tradeoff is that among winning root moves the
solver returns an arbitrary one instead of the spread-maximal one, and a
proven-lost position (`value <= kFirstWinAlpha`) is meaningless to compare
move-by-move, so `EndgameHastyBotAgent` discards it and plays HastyBot's
static-equity move instead -- final spread still matters to the game log even
in a lost position.

At a fixed `--endgame-nodes` budget, cost is governed by the budget itself
(iterative deepening spends the budget regardless of window width), so wld
mode's wall-time cost roughly tracks spread mode's at the same budget:

| `--endgame-nodes` | spread-mode cost | wld-mode cost |
|---|---|---|
| 50 | 1.07x | 1.04x |
| 100 | 1.26x | 1.22x |
| 200 | 1.97x | 1.89x |
| 400 | 6.13x | 5.90x |
| 800 | 16.70x | 16.26x |
| 1600 | 29.72x | 31.33x |

(`--mode=games --games=200 --seed=7 --threads=1`, with and without `--wld`;
single-shard, illustrative rather than a strength-grade sample.)

Strength, pooling 6 seed shards x 800 games = 4800 games per budget (same
paired-seat protocol as the spread-mode table above):

| `--endgame-nodes` | mean eg spread (wld) | W | D | L |
|---|---|---|---|---|
| 200 | +0.20 +/- 0.02 pts/game | 2398 | 17 | 2385 |
| 400 | +0.47 +/- 0.06 | 2399 | 24 | 2377 |
| 1600 | +2.44 +/- 0.08 | 2471 | 17 | 2312 |

For comparison, spread mode at the same budgets: 200 = 2.00x/+0.40 +/- 0.03,
400 = 4.5x/+0.85 +/- 0.07, 1600 = 21x/+4.96 +/- 0.19 (from the table above).

Under wld the mean spread is **expected to shrink by construction**: the
solver stops maximizing point margin once it has proven a win, so it no
longer plays for the same spread among winning lines. The number to judge wld
strength by is the **W/D/L record**, not the spread column. By that measure
wld mode still beats plain HastyBot at every budget (win rate exceeds loss
rate by +0.3pp at 200, +0.5pp at 400, +3.3pp at 1600 games), with the same
qualitative budget-dependence as spread mode -- the margin over greedy play
grows with budget -- but a visibly smaller edge than spread mode's win/loss
margin would likely show at the same budget, since wld mode's own root-move
choice among wins is unoptimized for spread and thus for many wins-that-stay-
wins style positional pressure. Budget 1600 remains the clearest win; the
sub-1% margins at 200 and 400 should not be read as more than "does not
lose ground," given the shard-level standard error is itself on that order.

## Reproducing

```
target/engine/endgame_bench --mode=games  --games=800 --seed=S --budgets=B --threads=1 [--wld]
target/engine/endgame_bench --mode=solves --games=30  --seed=42 --budgets=100,200,400,1600,5000,20000
```

For strength numbers, run several `--seed` shards per budget and pool them;
single-shard head-to-head spreads carry a few points of standard error even
with seat mirroring. `--wld` applies to `--mode=games` only; it makes both the
endgame-vs-endgame cost sweep and the endgame-vs-hasty head-to-head use the
first-win window.
