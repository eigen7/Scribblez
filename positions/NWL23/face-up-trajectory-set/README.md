# Face-up trajectory position set

Hand-maintained positions for watching the evidence loop think: each `.gcg`
is a decision point whose sim evidence should move the evidence-conditioned
model off its plain first pass. The dashboard's trajectory pane runs the
trajectory recipe here (anchor, proposer picks, uniform tail, all CRN-simmed)
and shows the observed-vs-predicted residual per simmed candidate and the
model's re-scoring at every evidence prefix; the evidence trainer reads the
same sidecars for its position-set metric.

## Convention

A file **is** the position to analyze: every move that led there is recorded
and nothing after it, the side to move is whoever acts next, and that side's
rack is given by a `#Rack1` / `#Rack2` pragma (engine: `sim/gcg_decision.h`;
`manual_gcg_tool` writes these). Under face-up leaves the opponent's public
leave is derived from their last event line (rack minus tiles played), so a
position with an interesting known leave needs the opponent's last move
recorded with its rack. The bag must be non-empty (the sims need it).

Sidecars are not committed: they depend on the proposer and recipe and are
generated on demand under `<mount>/cache/trajectory_sets/<set>/` by
`scribblez.sim_evidence.position_sets.ensure_sobs`, or by hand:

    ./target/engine/evidence_trajectory_generator --gcg-dir positions/NWL23/face-up-trajectory-set \
        --out-dir /tmp/traj --model <student.onnx> --open-leaves

## Positions

- `egotize-lane.gcg` — Hasty_1 to move with AEEGSTV, 440–387, Hasty_2 just
  bingoed INCASED. HastyBot plays E11 GAVE, opening 13G to EGOTIZE, which
  Hasty_2 draws with probability ~8%. The plain model under-reads the lane;
  one sim of GAVE observes opp-play-and-win mass there. The exhibit: does the
  conditioned re-scoring lift a lane-blocking alternative? (The position
  before the last move of `position-eval-test-dataset/pos-6.gcg`.)
