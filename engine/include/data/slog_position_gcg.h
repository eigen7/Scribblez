#pragma once

// A .slog position exported as GCG, for viewing and for the examples
// directories: the game through the position's played move, named so the turn
// can be found again (neural_rank_tool --turn, the web UI's GCG loader).

#include "data/slog_sampling.h"

#include <ostream>
#include <string>
#include <vector>

namespace scribblez {

// Game `at.game_idx` of the loaded .slog `slog` through turn `at.turn_idx`'s
// played move, as GCG with `notes` in its header. The scores stop at that
// turn, so the writer emits no end-of-game rack lines for the unfinished game.
void write_slog_position_gcg(const std::vector<char>& slog, const binlog::GamePositionIndex& at,
                             std::vector<std::string> notes, std::ostream& out);

// "<stem>-g<game>-turn<N>.gcg", N the turn's 1-based number.
std::string slog_position_gcg_name(const std::string& stem, const binlog::GamePositionIndex& at);

}  // namespace scribblez
