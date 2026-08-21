# Test data

Every fixture a test reads lives here, and a test reads nothing else. The
directory reaches C++ tests as the `SCRIBBLEZ_TEST_DATA_DIR` compile
definition (see `engine/CMakeLists.txt`) and Python tests as a path built from
the repo root.

The rule exists because the working sets under `positions/` are *live data*:
they get re-harvested, renamed and re-shaped whenever the dataset behind them
is regenerated. A test that reads one is asserting against a moving target.
The regeneration in `cf50ebc1` renamed `pos-N.gcg` to `pos-NN.gcg` and dropped
a trailing move, which broke three `MonteCarloSimTest` cases outright and
silently turned six Python tests into false skips.

A fixture here is frozen: copy the file in and assert against it. Never point
a test at `positions/`, and never quietly `skip()` when a fixture is missing --
these files are committed, so an absent one is a bug, not a reason to pass.

## The .gcg fixtures

Both `.gcg` conventions appear here; which one a file follows is what its
reader expects, not a property of the directory.

| File | Convention | What it is |
| --- | --- | --- |
| `FOE.gcg` | endgame (`read_gcg_endgame`) | Alice, down 141 with AABCGNT, cannot block Bob's FOE out-plays nor outscore them: a cheap proven loss. |
| `postbingo-gave.gcg` | post-move (`read_gcg_post_move`) | Hasty_2 bingoed INCASED, Hasty_1 answered E11 GAVE. The final mover is the POV and the opponent kept nothing, so the hidden- and face-up-leave conditions coincide. Truncating its last two moves gives a large known leave (ACEINS). |
| `egotize-lane.gcg` | decision point (`read_gcg_position`) | The same game one move earlier -- Hasty_1 to move with AEEGSTV, 440-387 -- as the trajectory pane reads it. |
| `ole.gcg`, `violets.gcg` | post-move | Two unremarkable midgame positions; the trajectory generator's two-position `.gcg` set. |
| `boreal.gcg` | post-move | Carries no `#RackN` pragma, which is what makes it the "a position set needs the mover's rack" rejection case. |
| `masked-racks.gcg` | post-move | Carries partially-known `#Rack1`/`#Rack2` pragmas (`_CE__MR`); the position-eval encoder's arm round-trip. |

All but `FOE.gcg` are frozen copies of positions that were live under
`positions/NWL23/` when the tests asserting on them were written.
