# Known positions

The positions the project is judged on: each one has an answer that a strong
player, or a careful reading of a long sim, arrives at and that static equity
does not. The aim is an agent that is not outplayed at any of them.

`positions.json` lists them. Each entry names:

- `gcg`: the game up to the position. Its last move is the one made there,
  so the position is the one before it, with that move's player to move.
- `reference`: the move or moves the analysis favours, written as a GCG
  writes a move and as the pick tool reports one (`8H .....ER` plays ER
  through five tiles on the board, `-EELLT` exchanges, `-` passes); empty
  where it favours none.
- `reference_from`: whose judgement that is.
- `static_equity_best`: HastyBot's move there.
- `sim_artifact`: moves a HastyBot-reply sim prefers for a reason that does
  not survive a thinking opponent. An agent that picks one has been fooled.
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
- `aceta`: [interesting-positions](../interesting-positions/README.md), whose
  two variants differ only in the opponent's hidden leave.
- The rest: the worked entries of
  [docs/analysis/hidden_blind_spots.md](../../../docs/analysis/hidden_blind_spots.md),
  copied from the `hidden_blind_spots` tag `v1`, which lives on the mount.
  Its opening-move pass (ACDEINT) is left out: it is that page's least
  certain entry, and a record in which one player has not yet moved does not
  convert.
