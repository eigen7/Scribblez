// Converts .gcg games into one .slog, so the .slog tools can run on
// hand-built positions (SupremeBot M1a's exhibits, docs/plans/supreme_bot_m1a.md).
//
//   gcg_to_slog --gcg a.gcg --gcg b.gcg --out-dir DIR [--face-up]
//
// A GCG records the rack each turn was played from, not the tiles drawn after
// it. A turn's draw is the player's next rack less what they kept; the draw
// after a player's last recorded turn, which no later rack shows, is filled
// from the tiles still unseen, in tile order, so every replayed bag count is
// right. Under face-up leaves only the kept tiles are public, so a filled draw
// changes nothing a face-up consumer reads. Each game is replayed after the
// draws are filled, and a GCG whose racks do not follow from its moves and
// draws is refused.

#include "data/binary_log.h"
#include "data/gcg_reader.h"
#include "game/bag.h"
#include "game/game_log.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace scribblez;

struct Options {
  std::vector<std::string> gcg_files;
  std::string out_dir;
  bool face_up = false;
};

std::string read_text(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw util::CleanException("cannot open {}", path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

Rack leave_of(const TurnRecord& t) {
  Rack leave = t.rack_before;
  for (int i = 0; i < t.move.num_glyphs(); ++i) leave.remove(t.move.glyph(i).rack_tile());
  return leave;
}

// `whole` less the tiles of `part`, which must all be on it.
Rack rack_minus(Rack whole, const Rack& part, const std::string& path) {
  for (const Tile t : part.tiles()) {
    if (t.is_empty()) break;
    if (!whole.remove(t)) {
      throw util::CleanException("{}: rack {} does not hold the kept tiles {}", path,
                                 whole.to_string(), part.to_string());
    }
  }
  return whole;
}

// The rack each player next plays from after turn `k`, if the GCG has one.
std::optional<Rack> next_rack(const std::vector<TurnRecord>& turns, size_t k) {
  for (size_t j = k + 1; j < turns.size(); ++j)
    if (turns[j].player == turns[k].player) return turns[j].rack_before;
  return std::nullopt;
}

// Fills every turn's draw: from the player's next rack where the GCG has one,
// else from `unseen` (the tiles no rack or draw accounts for), as many as the
// move draws from the bag it saw.
void fill_draws(std::vector<TurnRecord>& turns, TileCounts unseen, const std::string& path) {
  std::vector<bool> known(turns.size());
  for (size_t k = 0; k < turns.size(); ++k) {
    if (const std::optional<Rack> next = next_rack(turns, k)) {
      turns[k].drawn = rack_minus(*next, leave_of(turns[k]), path);
      unseen.remove(turns[k].drawn.counts());
      known[k] = true;
    }
  }
  int bag = Bag::kTotalTiles - 2 * RACK_SIZE;
  for (size_t k = 0; k < turns.size(); ++k) {
    TurnRecord& t = turns[k];
    const bool play = t.move.type() == MoveType::PLAY;
    if (!known[k]) {
      const int count = play ? std::min(t.move.num_glyphs(), bag) : t.move.num_glyphs();
      Rack drawn;
      for (int letter = 0; letter < TILE_KINDS && drawn.size() < count; ++letter) {
        const Tile tile = Tile::of(uint8_t(letter));
        while (drawn.size() < count && unseen.remove(tile)) drawn.add(tile);
      }
      t.drawn = drawn;
    }
    if (play) bag -= t.drawn.size();
  }
}

GameLogStorage game_log_of(const ParsedGcgGame& game, const std::string& path) {
  GameLogStorage log;
  log.player_names = game.player_names;
  for (const ParsedGcgTurn& t : game.turns) log.turns.push_back(t.record);
  for (int p = 0; p < 2; ++p) {
    const auto first = std::ranges::find(log.turns, p, &TurnRecord::player);
    if (first == log.turns.end()) throw util::CleanException("{}: player {} never moves", path, p);
    log.initial_racks[size_t(p)] = first->rack_before;
  }
  TileCounts unseen = TileCounts::full_distribution();
  for (const Rack& r : log.initial_racks) unseen.remove(r.counts());
  fill_draws(log.turns, unseen, path);

  const std::vector<TurnRecord> recorded = log.turns;
  binlog::ReplayStart start;
  start.racks = log.initial_racks;
  start.bag_size = Bag::kTotalTiles - 2 * RACK_SIZE;
  start.first_player = log.turns.front().player;
  binlog::replay_turn_records(start, log.turns.data(), int(log.turns.size()));
  for (size_t k = 0; k < log.turns.size(); ++k) {
    if (!(log.turns[k].rack_before == recorded[k].rack_before)) {
      throw util::CleanException("{}: turn {} replays from rack {}, but the GCG records {}", path,
                                 k + 1, log.turns[k].rack_before.to_string(),
                                 recorded[k].rack_before.to_string());
    }
  }
  log.final_scores = log.turns.back().cumulative_scores;
  return log;
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("gcg_to_slog options");
    desc.add_options()("help,h", "show this help and exit")(
      "gcg", po::value<std::vector<std::string>>(&opt.gcg_files)->required(),
      "a .gcg game to convert (repeatable); each becomes one game of the .slog")(
      "out-dir", po::value<std::string>(&opt.out_dir)->required(), "where the .slog is written")(
      "face-up", po::bool_switch(&opt.face_up), "mark the games as played with face-up leaves");
    util::parse_command_line(argc, argv, desc);
    std::filesystem::create_directories(opt.out_dir);
    binlog::BinaryLogWriter writer(opt.out_dir, int(opt.gcg_files.size()),
                                   opt.face_up ? binlog::kFlagFaceUpLeaves : 0);
    for (const std::string& path : opt.gcg_files) {
      ParsedGcgGame game;
      std::string error;
      if (!read_gcg_text(read_text(path), &game, &error))
        throw util::CleanException("{}: {}", path, error);
      writer.append(game_log_of(game, path));
      std::cerr << path << ": " << game.turns.size() << " turns\n";
    }
    writer.flush();
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
