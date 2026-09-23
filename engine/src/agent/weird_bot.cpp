#include "agent/weird_bot.h"

#include "agent/agent_options.h"
#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/move.h"
#include "game/tile.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"

#include <boost/program_options.hpp>

#include <string>
#include <vector>

namespace scribblez {

namespace {

// The forcing rule, applied each turn:
//   1. With no tracked leave, play HastyBot's greedy move.
//   2. Take T, the highest-value non-blank tile in the leave.
//   3-4. Find the empty square where T forms the highest-scoring perpendicular
//        word.
//   5. Play the highest-equity legal play that puts T on that square along the
//      target axis.
//   6. Record the tiles this move keeps as the new leave.
// Any step that finds nothing falls back to HastyBot's greedy move.

// The empty-tile sentinel doubles as "the leave holds no non-blank tile".
constexpr Tile kNoTile = Tile::empty();

// Step 2. Ties go to the lowest letter (a Rack is sorted). Blanks never
// force; an all-blank or empty leave yields kNoTile.
Tile highest_forcing_tile(const Rack& leave) {
  Tile best = kNoTile;
  int best_value = 0;
  for (int i = 0; i < leave.size(); ++i) {
    const Tile t = leave.tiles()[i];
    if (t.is_blank() || t.is_empty()) continue;
    if (best == kNoTile || t.value() > best_value) {
      best = t;
      best_value = t.value();
    }
  }
  return best;
}

// The score of the perpendicular word formed by placing a tile worth
// `tile_value` on the empty square (r, c). Only that square's premiums apply.
int cross_word_score(const Board& board, int r, int c, const CrossCheck& cc, int tile_value) {
  const Premium p = board.premium_at(r, c);
  return (cc.score + tile_value * p.letter_mult()) * p.word_mult();
}

// The square chosen in step 4.
struct ForcingTarget {
  int r = 0;
  int c = 0;
  bool transposed = false;  // false: T sits in a horizontal main word; true: vertical
  bool found = false;
};

// Steps 3-4, over both orientations of every empty square. A square must
// admit T in its cross-check and have a perpendicular neighbour, since a
// vacuous cross-check forms no word. Ties go to the lowest (r, c), then
// orientation. Board::cross_check_at() reads the dictionary that
// ensure_movegen_caches() binds, so that must run first.
ForcingTarget best_cross_check_square(const Board& board, Tile t) {
  ForcingTarget best;
  int best_score = 0;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!board.at(r, c).is_empty()) continue;
      for (int axis = 0; axis < 2; ++axis) {
        const bool transposed = axis == 1;
        // View coordinates: transposed swaps row and column (see Board).
        const int vr = transposed ? c : r;
        const int vc = transposed ? r : c;
        const CrossCheck cc = board.cross_check_at(transposed, vr, vc);
        if (!cc.has_neighbor) continue;
        if ((cc.mask & (1u << t.index())) == 0) continue;
        const int score = cross_word_score(board, r, c, cc, t.value());
        if (!best.found || score > best_score) {
          best = {r, c, transposed, true};
          best_score = score;
        }
      }
    }
  }
  return best;
}

// Whether play `m` places a new, non-blank `t` on (want_r, want_c). Walks the
// lane as Board::apply() does: the i-th set square carries the i-th glyph.
bool play_forces_tile(const Move& m, int want_r, int want_c, Tile t) {
  if (m.type() != MoveType::PLAY) return false;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  int gi = 0;
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const Glyph g = m.glyph(gi++);
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    if (r == want_r && c == want_c) return g.has_letter() && !g.is_blank() && g.letter() == t;
  }
  return false;
}

// Step 5; nullptr when no legal play qualifies.
const Move* best_forcing_play(const std::vector<Move>& plays, const ForcingTarget& tgt, Tile t,
                              const MoveRequest& req) {
  const bool want_horizontal = !tgt.transposed;
  const HastyEquity& eq = HastyEquity::instance();
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);
  const Move* best = nullptr;
  double best_eq = 0.0;
  for (const Move& m : plays) {
    if (m.horizontal() != want_horizontal) continue;
    if (!play_forces_tile(m, tgt.r, tgt.c, t)) continue;
    const double e = eq.equity(m, req.board, req.bag_size, req.opp_rack, leaves);
    if (best == nullptr || hasty_move_better(e, m, best_eq, *best)) {
      best = &m;
      best_eq = e;
    }
  }
  return best;
}

// Step 6. A move's glyphs are exactly the rack tiles it consumes for every
// move type (placed tiles, surrendered tiles, or none for a pass), so one loop
// covers all three.
Rack leave_after_move(const Rack& rack, const Move& move) {
  Rack leave = rack;
  const int n = move.num_glyphs();
  for (int i = 0; i < n; ++i) leave.remove(move.glyph(i).rack_tile());
  return leave;
}

namespace po = boost::program_options;

}  // namespace

WeirdBotAgent::WeirdBotAgent(int thread_id, const std::string& name) : Agent(thread_id, name) {}

void WeirdBotAgent::begin_game(const BeginGameRequest&) { leave_ = Rack{}; }

Move WeirdBotAgent::choose_move(const MoveRequest& req) const {
  if (leave_.empty()) return hasty_best_move_wmp(req);

  const Tile t = highest_forcing_tile(leave_);
  if (t == kNoTile) return hasty_best_move_wmp(req);

  req.board.ensure_movegen_caches(req.dict);
  const ForcingTarget tgt = best_cross_check_square(req.board, t);
  if (!tgt.found) return hasty_best_move_wmp(req);

  const std::vector<Move> plays = generate_legal_plays(req);
  const Move* forced = best_forcing_play(plays, tgt, t, req);
  return forced ? *forced : hasty_best_move_wmp(req);
}

MoveDecision WeirdBotAgent::make_move(const MoveRequest& req) {
  const Move move = choose_move(req);
  leave_ = leave_after_move(req.my_rack, move);
  return move;
}

std::unique_ptr<WeirdBotAgent> WeirdBotAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name) {
  // Parse against an empty option set so any token is an error, not ignored.
  po::options_description desc;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=weirdbot options: {}", e.what());
  }

  // The forcing-play ranking and the fallback both read the equity tables.
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  return std::make_unique<WeirdBotAgent>(thread_id, name);
}

std::string WeirdBotAgent::options_help() {
  po::options_description desc;  // no options; the description carries the help
  return agent_options_help(
    "  Diagnostic self-play bot: forces its highest-value retained leave tile\n"
    "  onto its best cross-check square, falling back to HastyBot when it has\n"
    "  no leave or no legal forcing play. Takes no options.\n",
    desc);
}

}  // namespace scribblez
