# Test data

Every fixture a test reads lives here, and a test reads nothing else. The
directory reaches C++ tests as the `SCRIBBLEZ_TEST_DATA_DIR` compile
definition (see `engine/CMakeLists.txt`) and Python tests as a path built from
the repo root.

The working sets under `positions/` are live data: they are re-harvested,
renamed and reshaped whenever the dataset behind them is regenerated, so a test
that reads one asserts against a moving target. A regeneration that renames a
file or drops a trailing move breaks such tests outright, or worse, turns them
into silent skips.

A fixture here is frozen: copy the file in and assert against it. Never point
a test at `positions/`, and never quietly skip when a fixture is missing. These
files are committed, so an absent one is a bug, not a reason to pass.

## The .gcg fixtures

Both `.gcg` conventions appear here; which one a file follows is what its
reader expects, not a property of the directory.

| File | Convention | What it is |
| --- | --- | --- |
| `FOE.gcg` | endgame (`read_gcg_endgame`) | Alice, down 141 with AABCGNT, can neither block Bob's FOE out-plays nor outscore them: a cheap proven loss. |
| `postbingo-gave.gcg` | post-move (`read_gcg_post_move`) | Hasty_2 bingoed INCASED, Hasty_1 answered E11 GAVE. The final mover is the POV and the opponent kept nothing, so the hidden- and face-up-leave conditions coincide. Truncating its last two moves gives a large known leave (ACEINS). |
| `egotize-lane.gcg` | decision point (`read_gcg_position`) | The same game one move earlier -- Hasty_1 to move with AEEGSTV, 440-387 -- as the trajectory pane reads it. |
| `ole.gcg`, `violets.gcg` | post-move | Two unremarkable midgame positions; the trajectory generator's two-position `.gcg` set. |
| `boreal.gcg` | post-move | Carries no `#RackN` pragma, which is what makes it the "a position set needs the mover's rack" rejection case. |
| `pos09-gnu.gcg` | decision point (`read_gcg_position`) | Position pos-09 of `positions/NWL23/position-eval-test-dataset`: the opponent holds G, and G on M7 forms GNU with the existing NU. `WeirdBot.ForcesGAtM7OnPos09` checks the forcing end to end on it. |
| `masked-racks.gcg` | post-move | Carries a partially known `#Rack1` pragma (`_CE__MR`), for the position-eval encoder's input-arm round trip. |

All but `FOE.gcg` are frozen copies of positions that were live under
`positions/NWL23/` when the tests asserting on them were written. Every `.gcg`
here also feeds `FootprintMaskSoundness`, which replays each game and checks
that the footprint masks admit every move played, so a new fixture is covered
there once it is added to that test's list.
