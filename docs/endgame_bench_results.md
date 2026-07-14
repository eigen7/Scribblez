# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, single-threaded runs, NWL23 lexicon,
Release build). They characterize the `hastybot-endgame` agent's cost and
strength as a function of its `--endgame-nodes` budget and its
`--endgame-objective` (see `EndgameObjective` in endgame_solver.h), and are the
basis for the shipped defaults (budget 220, lexicographic objective).

**Hardware**: Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside
the project's Docker container, Release build.

## The objectives, and what each is for

- **lexicographic** (agent default): prove the win/draw/loss class first (a
  narrow-window pass capped at half the budget), then spend the rest
  maximizing spread, playing the spread move only when it provably preserves a
  proven class. Never trades the class for points; the right objective for
  games played to their end.
- **first-win**: the break-out objective for self-play generation. It only
  resolves the class and stops the moment the class is proven, exposing the
  proof as `EndgameResult::proven_class` -- a proven class means the rest of
  the game is not worth computing, and the saved budget belongs to other
  games. Among winning moves it returns an arbitrary one.
- **spread**: the pure margin objective, full window for the exact final
  spread. Exact spread play is class-optimal by construction (a positive final
  spread IS a win); its class risk is confined to unproven estimates.

## Methodology

- **Cost** (`--mode=games`): mean wall-time per game of endgame-vs-endgame
  self-play, as a ratio to the hasty-vs-hasty baseline (~5 ms/game, ~215
  games/s single-threaded), same seeds across configs. Cost rows come from
  dedicated single runs on an otherwise idle machine (`--games=200 --seed=7
  --threads=1`); run-to-run baseline drift is ~2%, and wall-clock ratios from
  parallel or loaded runs are not comparable.
- **Strength** (`--mode=games` head-to-head): the endgame bot's record against
  a plain greedy HastyBot. Each seed is played twice with the seats mirrored
  and the pair pooled, which cancels per-seed tile-draw luck -- the dominant
  variance source; unpaired estimates are not usable. Strength rows pool 6
  independent 800-game seed shards (4800 games per cell); quoted uncertainty
  is the standard error across shards.
- **Bucketing by bag-empty spread**: every analysis is conditioned on the
  absolute score spread when the bag empties, because the two things worth
  measuring live in opposite buckets. *Decision accuracy* only shows up in
  close endgames -- in decided games every objective converts equally, so
  their games contribute pure noise to a win/loss record (measured below:
  W-L +0.0pp in the 60+ bucket at budget 220, for every objective, 2608
  games). *Break-out efficiency* -- prove the decided game's class fast and
  hand the budget back -- is only meaningful in those same decided games.
  Head-to-head games are bucketed by the seed's *baseline hasty-vs-hasty*
  bag-empty spread, a deterministic conditioning variable that is identical
  across objectives, budgets, and seats; solves-mode positions are bucketed by
  their own decision-point spread.
- **Which metric judges what**: W-L margin in the small-spread buckets judges
  class play (what the lexicographic and first-win objectives protect); mean
  spread judges margin play and the value-target quality the training pipeline
  consumes; class-proof rate and %-of-budget-spent in the large-spread buckets
  judge the break-out imperative.

## Cost vs node budget, per objective

| `--endgame-nodes` | lexicographic | spread | first-win |
|---|---|---|---|
| 100 | 1.24x | 1.24x | 1.25x |
| **220 (default)** | **2.01x** | 2.13x | 2.11x |
| 400 | 5.28x | 5.89x | 5.87x |
| 1600 | 28.2x | 31.8x | 33.5x |

The lexicographic objective is the cheapest at every real budget: its class
pass proves the easy positions at the narrow window's price and warms the
transposition table for its spread pass. The default budget of 220 keeps the
default objective's endgame-vs-endgame games at ~2x the hasty-vs-hasty game
time.

## Strength vs greedy HastyBot (6x800 paired games per cell)

Budget 220:

| objective | mean spread | W-L overall | W-L, bag-empty spread 0-19 (800 games) |
|---|---|---|---|
| lexicographic | +0.43 +/- 0.04 | +0.50pp | +0.75pp |
| spread | +0.53 +/- 0.04 | +0.60pp | +1.38pp |
| first-win | +0.34 +/- 0.08 | +0.52pp | +1.12pp |

Budget 1600:

| objective | mean spread | W-L overall | W-L, bag-empty spread 0-19 (800 games) |
|---|---|---|---|
| lexicographic | +4.23 +/- 0.13 | +3.88pp | +14.00pp |
| spread | +4.81 +/- 0.14 | +4.33pp | +14.75pp |
| first-win | +2.20 +/- 0.16 | +4.02pp | +14.00pp |

Readings:

- **Class play is equal across objectives within noise** (the 0-19 bucket's
  W-L standard error is roughly +/-1.5pp at 800 games): under a hard node cap,
  proofs land where they land regardless of what the objective does with them,
  and the close games the objectives could disagree on are rarer still. The
  bucket structure itself is the loud signal: all of the win/loss value of
  endgame solving lives in the close bucket (+14pp at 1600), and none in the
  decided ones (+0.0-0.4pp) -- an unconditioned W-L record dilutes the signal
  by the bucket ratio.
- **The lexicographic insurance premium is ~0.1-0.6 pts/game of banked
  spread** (vs the spread objective) at equal-or-lower cost. What it buys is
  the guarantee -- enforced by test, not measured by these samples -- that a
  proven class is never traded for points.
- **first-win banks roughly half the spread** (arbitrary winning moves) at the
  same class play; as an agent objective for full games it is dominated, and
  its value is the break-out signal below.
- At high budgets the spread objective's margin play still owns the frontier
  it has always owned (~+5.6 pts/game plateau at budgets 5000-20000, measured
  under this objective).

## Break-out efficiency (first-win, `--mode=solves`, 256 positions)

Per (budget, |spread| bucket): class-proof rate and the share of the node
budget actually spent (what a self-play generator would NOT get back).

| budget | \|spread\| | positions | % class proven | % budget spent |
|---|---|---|---|---|
| 220 | 0-19 | 34 | 21% | 38% |
| 220 | 20-59 | 71 | 21% | 31% |
| 220 | 60+ | 151 | 27% | 30% |
| 1600 | 0-19 | 34 | 41% | 47% |
| 1600 | 20-59 | 71 | 44% | 59% |
| 1600 | 60+ | 151 | 41% | 58% |

At the default budget, a quarter of decided bag-empty positions prove their
class while spending ~30% of the cap; proofs deepen with budget (41% at
1600). This is the solver-side half of self-play break-out; the game-runner
half (terminating a generation game on `proven_class` and logging the proven
result) is not built yet.

## Per-solve accuracy view (lexicographic, `--mode=solves`, 256 positions)

| budget | \|spread\| | mean us | mean depth | % differ | % class | % value proven |
|---|---|---|---|---|---|---|
| 220 | 0-19 | 1529 | 0.4 | 32% | 18% | 15% |
| 220 | 20-59 | 1504 | 0.4 | 32% | 17% | 17% |
| 220 | 60+ | 1309 | 0.5 | 38% | 17% | 17% |
| 1600 | 0-19 | 46500 | 1.3 | 44% | 32% | 32% |
| 1600 | 20-59 | 88583 | 1.0 | 39% | 34% | 32% |
| 1600 | 60+ | 45206 | 1.2 | 42% | 37% | 34% |

"% differ" counts solves whose move differs from greedy HastyBot's. The
lexicographic class rate runs slightly below the first-win table's: its class
pass is capped at half the budget (the reserved half guarantees the margin
fallback is a real search; an uncapped class pass returns margin-blind moves
on every position it fails to prove, which measurably forfeits spread).

## Reproducing

```
target/engine/endgame_bench --mode=games  --games=800 --seed=S --budgets=B --threads=1 \
    --objective=lexicographic|first-win|spread [--spread-buckets 20,60]
target/engine/endgame_bench --mode=solves --games=100 --seed=42 --budgets=220,1600 \
    --objective=... [--no-futility]
```

For strength numbers, run several `--seed` shards per budget and pool them;
single-shard head-to-head spreads carry a few points of standard error even
with seat mirroring, and W-L margins in the close bucket need pooled shards to
mean anything. For cost numbers, run one config at a time on an idle machine.
`--no-futility` (solves mode) A/Bs the opponent-outplay futility pruning; on a
fixed 60-position batch it cuts ~49% of nodes and lifts the proof-rate columns
(see EndgameSolver.OutplayFutilityCutsNodes for the in-tree guard).
