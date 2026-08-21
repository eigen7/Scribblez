#include "agent/weird_bot.h"

#include "agent/agent_options.h"
#include "agent/macondo_bot.h"
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

// The empty-tile sentinel doubles as "the leave holds no non-blank tile".
constexpr Tile kNoTile = Tile::empty();

// Step 2: the highest face-value non-blank tile in `leave`, ties broken by
// lowest letter index. Blanks (value 0, wild) never force, so they are skipped;
// an all-blank or empty leave yields kNoTile.
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

// The perpendicular cross-word score of placing a value-`tile_value` tile on the
// empty square (r, c): the run's existing tile values (cc.score) plus the new
// tile scored under the square's letter premium, the whole word taken under its
// word premium. Only the newly covered square's premiums apply.
int cross_word_score(const Board& board, int r, int c, const CrossCheck& cc, int tile_value) {
  const Premium p = board.premium_at(r, c);
  return (cc.score + tile_value * p.letter_mult()) * p.word_mult();
}

// The winning cross-check square for tile T, chosen in step 4.
struct ForcingTarget {
  int r = 0;
  int c = 0;
  bool transposed = false;  // false: T sits in a horizontal main word; true: vertical
  bool found = false;
};

// Steps 3-4: over both orientations and every empty square, the square whose
// perpendicular cross-word (with T placed) scores highest. A candidate must
// admit T in its cross-check and form a real perpendicular word (has_neighbor);
// a vacuous all-letters cross-check forms no word and never wins. Ties break by
// lowest (r, c) then orientation, achieved by iterating (r, c, transposed) in
// ascending order and replacing only on a strictly higher score.
//
// Cross-checks are read straight from Board::cross_check_at(), which computes a
// square's cross-check on demand from the live board; ensure_movegen_caches()
// must have run first to bind the dictionary it reads.
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

// Whether play `m` places a newly-placed, non-blank tile equal to `t` on the
// square (want_r, want_c). Mirrors Board::apply()'s lane walk so the i-th set
// square carries the i-th stored glyph.
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

// Step 5: among the legal plays, the highest-hasty-equity one that runs along
// the target axis and places T on the target square. nullptr when none exists.
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

// Step 6: my_rack minus the rack tiles the move consumed. A move's stored glyphs
// are exactly those tiles for every move type -- placed tiles for a play (a
// placed blank consumes the rack blank), surrendered tiles for an exchange, none
// for a pass -- so one loop covers all three.
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
  // Step 1: no tracked leave (first move, or the last move kept nothing) -> the
  // plain greedy hasty argmax fallback.
  if (leave_.empty()) return hasty_best_move_wmp(req);

  const Tile t = highest_forcing_tile(leave_);
  if (t == kNoTile) return hasty_best_move_wmp(req);  // an all-blank leave never forces

  req.board.ensure_movegen_caches(req.dict);
  const ForcingTarget tgt = best_cross_check_square(req.board, t);
  if (!tgt.found) return hasty_best_move_wmp(req);  // T fits no real cross-check square

  const std::vector<Move> plays = generate_legal_plays(req);
  const Move* forced = best_forcing_play(plays, tgt, t, req);
  return forced ? *forced : hasty_best_move_wmp(req);  // no legal forcing play -> fallback
}

MoveDecision WeirdBotAgent::make_move(const MoveRequest& req) {
  const Move move = choose_move(req);
  leave_ = leave_after_move(req.my_rack, move);
  return move;
}

std::unique_ptr<WeirdBotAgent> WeirdBotAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name) {
  // WeirdBot takes no options of its own; parse an empty set so any token is a
  // reported error rather than silently ignored.
  po::options_description desc;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=weirdbot options: {}", e.what());
  }

  // The forcing-play ranking and the fallback both read the process-wide equity
  // tables, so load the active lexicon's defaults now (idempotent).
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
