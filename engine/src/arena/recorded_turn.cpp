#include "arena/recorded_turn.h"

#include "data/binary_log.h"
#include "game/bag.h"
#include "util/exception.h"
#include "util/io.h"

#include <fstream>
#include <sstream>

namespace scribblez {

namespace {

// The game's state before its first turn.
binlog::ReplayStart start_of(const GameLog& g) {
  return {.racks = g.initial_racks,
          .scores = g.initial_scores,
          .bag_size = Bag::kTotalTiles - g.initial_racks[0].size() - g.initial_racks[1].size(),
          .first_player = 0};
}

void check_turn(const GameLog& g, int turn) {
  if (turn < 0 || turn >= g.num_records)
    throw util::CleanException("turn {} is outside the game's {} turns", turn, g.num_records);
}

// Throws unless `buf`, the bytes of the .slog at `path`, is a hidden-leaves
// file of this format version.
void check_slog(const std::string& path, const std::vector<char>& buf) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
  if (buf.size() < sizeof(binlog::FileHeader) || hdr->magic != binlog::kMagic ||
      hdr->version != binlog::kVersion)
    throw util::CleanException("{}: not a version-{} .slog", path, binlog::kVersion);
  if ((hdr->flags & binlog::kFlagFaceUpLeaves) != 0)
    throw util::CleanException("{} was played with face-up leaves; the replay hides them", path);
}

}  // namespace

std::vector<RecordedTurnRef> read_recorded_turn_refs(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw util::CleanException("cannot open {}", path);
  std::vector<RecordedTurnRef> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream fields(line);
    RecordedTurnRef ref;
    if (!(fields >> ref.slog >> ref.game >> ref.turn) || ref.turn < 0)
      throw util::CleanException("{}: expected \"<.slog path> <game> <turn>\", got \"{}\"", path,
                                 line);
    out.push_back(std::move(ref));
  }
  return out;
}

std::map<std::string, std::vector<char>> load_recorded_slogs(
  const std::vector<RecordedTurnRef>& refs) {
  std::map<std::string, std::vector<char>> out;
  for (const RecordedTurnRef& ref : refs) {
    auto it = out.find(ref.slog);
    if (it == out.end()) {
      std::vector<char> buf = util::read_file_bytes(ref.slog);
      check_slog(ref.slog, buf);
      it = out.emplace(ref.slog, std::move(buf)).first;
    }
    const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(it->second.data());
    if (ref.game >= hdr->num_games)
      throw util::CleanException("{}: game {} of {}", ref.slog, ref.game, hdr->num_games);
  }
  return out;
}

RecordedPosition recorded_position(const GameLog& g, int turn) {
  check_turn(g, turn);
  binlog::ReplayStart state = start_of(g);
  RecordedPosition out;
  for (int k = 0; k < turn; ++k) {
    TurnRecord rec = g.records[k];
    state = binlog::replay_turn_records(state, &rec, 1);
    out.board.apply(rec.move);
    out.scoreless_turns = rec.move.type() == MoveType::PLAY ? 0 : out.scoreless_turns + 1;
  }
  out.racks = state.racks;
  out.scores = state.scores;
  out.bag = state.bag_size;
  out.mover = state.first_player;
  return out;
}

RecordedPick pick_at_recorded_turn(Agent& agent, const GameLog& g, int turn,
                                   const Dictionary& dict) {
  check_turn(g, turn);
  const int seat = turn % 2;
  agent.begin_game({.initial_scores = g.initial_scores});
  binlog::ReplayStart state = start_of(g);
  Board board;
  const Rack hidden;
  for (int k = 0;; ++k) {
    const int mover = state.first_player;
    const Rack& opp_rack = state.bag_size == 0 ? state.racks[1 - mover] : hidden;
    const MoveRequest req{board,
                          dict,
                          state.racks[mover],
                          opp_rack,
                          state.scores[mover],
                          state.scores[1 - mover],
                          state.bag_size};
    if (k == turn) {
      return {.board = board,
              .rack = state.racks[mover],
              .score = state.scores[mover],
              .opp_score = state.scores[1 - mover],
              .bag = state.bag_size,
              .pick = agent.make_move(req).move};
    }
    if (mover == seat) agent.observe_turn(req);
    TurnRecord rec = g.records[k];
    state = binlog::replay_turn_records(state, &rec, 1);
    board.apply(rec.move);
    agent.observe_move(rec.move);
  }
}

}  // namespace scribblez
