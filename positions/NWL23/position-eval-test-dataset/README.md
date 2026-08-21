# Position-evaluation test set

The hand-built positions the dashboard's Positions tab scrubs: each model
generation's prediction against a committed Monte-Carlo ground truth. The
machine-harvested companion is `../position-eval-test-dataset-large/`.

## Convention

A file **is** a post-move position: every move that led there is recorded and
nothing after it, and the **final recorded move is the evaluated player's** --
the position is the board after it, evaluated from that player's POV holding
their leave (that move's rack minus what it placed). The other player acts
next; their rack is what their own last recorded move retained plus hidden
draws, so their last move line must carry its full rack (it defines the
leave). `#RackN` pragmas are not read here. The bag must be non-empty (the
sims need it). Engine reader: `read_gcg_post_move` in `data/gcg_post_move.h`;
`manual_gcg_tool` writes conforming files.

The convention does not require any particular shape. The set is regenerated
wholesale (files are renamed and re-shaped when it is), so nothing outside the
dashboard and the sim tool may depend on a file here: tests read frozen copies
under `engine/tests/data/` instead.

## Ground truth

`monte-carlo-sim-results.<condition>.json`, written by `monte_carlo_sim_tool`
(~10k EndgameHastyBot rollouts per position; W/L/D, the exact score-delta
histogram, and the placement planes), one file per information condition
(`engine/include/sim/monte_carlo_sim.h`):

- `face-up-leaves` -- every rollout seats the opponent with their retained
  leave;
- `hidden-leaves` -- the leave is inferred from their last move
  (`belief::RackInferrer`, the Macondo rangefinder port) and sampled per
  rollout; uniform after a bingo.

A tag is measured against the truth of its own `face_up_leaves` param. On a
position whose opponent kept nothing the two truths are identical.

Regenerate after adding positions:

    ./target/engine/monte_carlo_sim_tool --dataset-name position-eval-test-dataset
