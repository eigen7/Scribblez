#include "data/slog_position_gcg.h"

#include "data/binary_log.h"
#include "data/gcg_writer.h"
#include "lexicon/lexicon.h"

#include <format>

namespace scribblez {

void write_slog_position_gcg(const std::vector<char>& slog, const binlog::GamePositionIndex& at,
                             std::vector<std::string> notes, std::ostream& out) {
  std::vector<TurnRecord> scratch;
  GameLog g = binlog::make_game_view(slog.data(), at.game_idx, scratch, nullptr);
  g.num_records = int(at.turn_idx) + 1;
  binlog::complete_turn_records(g, g.num_records, scratch);
  g.final_scores = scratch[at.turn_idx].cumulative_scores;
  g.player_names = {"hasty0", "hasty1"};
  GcgWriteOptions gcg;
  gcg.lexicon_name = Lexicon::instance().name();
  gcg.notes = std::move(notes);
  write_game_log_gcg(g, out, gcg);
}

std::string slog_position_gcg_name(const std::string& stem, const binlog::GamePositionIndex& at) {
  return std::format("{}-g{}-turn{}.gcg", stem, at.game_idx, at.turn_idx + 1);
}

}  // namespace scribblez
