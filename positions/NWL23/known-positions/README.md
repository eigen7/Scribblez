# Known positions

The positions the project is judged on. A position belongs here when there
is good reason to believe that its best move is not among HastyBot's ten best
by static equity: a strong player made it and can say why, or a reading of
the position and a long sim agree on it. The aim is an agent that is not
outplayed at any of them.

Positions whose sim-best move is a pass are left out. A pass sims well
against an opponent who never passes back, which is the sim's opponent and
not a real one.

`positions.json` lists them. Each entry names:

- `gcg`: the game up to the position. Its last move is the one made there,
  so the position is the one before it, with that move's player to move.
- `reference`: the move or moves the analysis favours, written as a GCG
  writes a move and as the pick tool reports one (`8H .....ER` plays ER
  through five tiles on the board, `-EELLT` exchanges, `-` passes); empty
  where it favours none.
- `reference_equity_rank`: where static equity ranks that move among the
  legal moves, HastyBot's own being 1. Every one is past 10.
- `evidence`: the reason to believe it.
- `static_equity_best`: HastyBot's move there.
- `source`: where the position is worked through, and `origin`: the game it
  came from.

`python3 -m scripts.deep_bot.known_position_picks --player "<spec>"` asks an
agent for its move at every one and tabulates the picks.

## Adding a position

A game the engine played replays as it is: copy its GCG, cut after the move
at the position. A human game's record usually does not, because a rack
written down for the opponent lacks tiles the same player kept the turn
before. `py/tools/one_off/known_position_gcg.py game.gcg --ply N --out
<id>.gcg` cuts the game at ply N and makes each player's racks follow from
one another, leaving the moves, the scores and the position untouched.

## Where they came from

- `richards-eellt`: [docs/analysis/richards-johnson-exchange](../../../docs/analysis/richards-johnson-exchange/README.md).
- `richards-allee-yeti`: a hypothetical continuation of that game, with
  Nigel Richards playing B6 ALLEE instead of exchanging, and Mike Johnson
  then holding ADEIIST.
- `aceta`: [interesting-positions](../interesting-positions/README.md), whose
  two variants differ only in the opponent's hidden leave.
- `avert`, `diner`, `trey`, `anneal`, `hins`: picked by David from the
  `hidden_blind_spots` tag `v1`'s survey, copied from that tag, which lives
  on the mount.
- The rest: the worked entries of
  [docs/analysis/hidden_blind_spots.md](../../../docs/analysis/hidden_blind_spots.md),
  copied from the `hidden_blind_spots` tag `v1`. That page's three passes
  (CHORD, INDIGENE, ACDEINT) are the ones left out, and A(I)M, O(I) and
  exchange seven were dropped later; their GCGs remain.
