# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, single-threaded runs, NWL23 lexicon,
Release build). They characterize the `hastybot-endgame` agent's cost and
strength as a function of `EndgameSolver::Params` -- the per-solve node
`budget` (default 220) and `spread_matters` -- and are the basis for those
defaults.

**Hardware**: Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside
the project's Docker container, Release build.

## The two settings

- **spread_matters = false** (default): resolve only the win/draw/loss class
  and stop the moment it is proven, exposing the proof as
  `EndgameResult::proven_class` and a full proof-certificate line as the
  agent's projected continuation. The self-play break-out setting: a decided
  game is not worth more compute, and a projection-respecting game loop plays
  the certificate out instead of prompting the agents. It presumes such a
  loop -- among winning (or losing) moves it plays a proof-arbitrary one, so
  in a loop that ignores projections it concedes margin it never tries to
  keep.
- **spread_matters = true**: never trade the class for points. A class pass
  capped at half the budget, then a spread pass on the remainder whose move is
  played only when it provably preserves a proven win or draw; a proven loss
  makes the spread pass pure defense; with no proof, margin play is the
  class-robust fallback. The setting for games played to their end.

Every class proof carries a valid certificate: the class-critical side's
moves come from fresh narrow-window re-proofs at each position of the walk
(over the warm table, outside the node budget, cost tracked in
`EndgameResult::certificate_nodes` and negligible in practice), and the line
is gated on replaying to the proven class. The test suite asserts a
certificate for every proven solve whose chosen move does not itself end the
game, and curated GCG endgames under engine/tests/data/ pin known positions
to their proven class and proof cost (see EndgameGcgCases).

## Methodology

- **Endgame-phase cost** (`--mode=endgames`): play N HastyBot-vs-HastyBot
  games, timing each whole game (A) and its endgame phase (B: from the first
  bag-empty decision to the end); then play each captured endgame out with
  solver agents on both seats, projections respected, timing the whole
  endgame (E). E/B is the endgame-phase multiplier, (A-B+E)/A the whole-game
  (self-play throughput) multiplier.
- **Whole-game cost** (`--mode=games`): mean wall-time per game of
  endgame-vs-endgame self-play (projections respected, as generation runs),
  as a ratio to the hasty-vs-hasty baseline (~5 ms/game, ~215 games/s
  single-threaded), same seeds across configs. Cost rows come from dedicated
  single runs on an otherwise idle machine; run-to-run baseline drift is ~2%.
- **Strength** (`--mode=games` head-to-head): the endgame bot's record against
  a plain greedy HastyBot, each seed played twice with the seats mirrored and
  the pair pooled (unpaired estimates are not usable). Every game respects
  projections, as self-play generation does, so once the bot proves a class
  the recorded spread is the certificate line's -- the margins the training
  pipeline actually sees. The W/D/L record is proof-invariant, so it measures
  the same thing with or without projections. Strength rows pool 6
  independent 800-game seed shards; uncertainty is the across-shard standard
  error.
- **Bucketing by bag-empty spread** (`--spread-buckets`): decision accuracy
  only shows up in close endgames -- in decided games every configuration
  converts equally, so their games contribute pure noise to a win/loss record
  (measured below: W-L +0.0pp in the 60+ bucket at budget 220, 2608 games) --
  while break-out throughput matters most exactly there. Head-to-head games
  are bucketed by the seed's *baseline hasty-vs-hasty* bag-empty spread, a
  deterministic conditioning variable identical across configurations and
  seats.
- **Which metric judges what**: W-L margin in the small buckets judges class
  play; mean spread judges margin play (and the value-target currency the
  training pipeline consumes) -- but only under spread_matters, since the
  break-out setting deliberately stops maximizing it; the endgames-mode
  multipliers judge what the solver does to self-play throughput.

## What the solver does to self-play throughput (`--mode=endgames`, 300 games)

Baseline: A ~= 4.4 ms/game, of which the endgame phase is B ~= 0.47 ms (~11%).
"all" bucket rows, spread_matters=0:

| budget | solver endgame ms (E) | endgame multiplier E/B | whole-game multiplier |
|---|---|---|---|
| 100 | 2.1 | 4.5x | 1.37x |
| **220 (default)** | 6.0 | 12.8x | **2.25x** |
| 400 | 22.5 | 48x | 5.98x |
| 1600 | 153 | 326x | 35.4x |

The whole-game multiplier is the self-play throughput number to keep tabs on:
at the default budget the solver makes the endgame phase ~13x more expensive,
which alone multiplies whole-game cost by ~2.25x. The games-mode sweep agrees
(same seeds, hasty baseline 1.00x): 100 -> 1.25x, 220 -> 2.11x, 400 -> 5.93x,
1600 -> 33.7x.

## Strength vs greedy HastyBot (6x800 paired games per cell)

| setting | budget | mean spread | W-L overall | W-L, bag-empty spread 0-19 (800 games) |
|---|---|---|---|---|
| spread_matters=1 | 220 | +0.44 +/- 0.04 | +0.50pp | +0.75pp |
| spread_matters=0 | 220 | +0.41 +/- 0.08 | +0.52pp | +1.12pp |
| spread_matters=1 | 1600 | +4.26 +/- 0.13 | +3.88pp | +14.00pp |
| spread_matters=0 | 1600 | +3.28 +/- 0.18 | +4.02pp | +14.00pp |

Readings:

- **Class play is identical across the two settings** (equal W/D/L within
  noise at both budgets; the close bucket's standard error is roughly
  +/-1.5pp at 800 games), and all of the win/loss value of endgame solving
  lives in that close bucket (+14pp at 1600 vs +0.0-0.4pp in decided games).
- **Certificates carry most of the margin for the break-out setting**: its
  post-proof play is proof-arbitrary, but the projected certificate line is
  proof-grade, so under projections it banks within a point of
  spread_matters=1 at 1600 -- at the narrow window's lower cost. The
  remaining gap is what the spread pass buys.
- The spread_matters margin play at high budgets is the same exact-endgame
  value it has always been (~+4.3 pts/game at 1600 at these seeds).

## Analyzing a single position

`endgame_tool --gcg FILE` solves one GCG endgame position (bag empty; the
mover's rack from a #RackN pragma, the opponent's derived from the board)
with a verbose trace of the machinery: the replier's out-play set, every root
move's block-or-outscore futility bound ("needs >= +X to reach a draw"), each
deepening iteration's verdict, the certificate walk, and the projected line.

## Reproducing

```
target/engine/endgame_bench --mode=endgames --games=300 --seed=42 --budgets=100,220,400,1600
target/engine/endgame_bench --mode=games    --games=200 --seed=7  --threads=1 --budgets=...
target/engine/endgame_bench --mode=games    --games=800 --seed=S  --threads=1 --budget=B \
    --spread-matters=0|1     # strength shards; pool several seeds S
target/engine/endgame_tool  --gcg engine/tests/data/FOE.gcg
```

For strength numbers, pool several seed shards; single-shard head-to-head
spreads carry a few points of standard error even with seat mirroring, and
close-bucket W-L margins need pooled shards to mean anything. For cost
numbers, run one config at a time on an idle machine.
