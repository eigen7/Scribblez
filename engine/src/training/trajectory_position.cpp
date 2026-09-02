#include "training/trajectory_position.h"

#include "agent/agent.h"
#include "data/gcg_writer.h"
#include "encoding/game_state_encoder.h"
#include "serve/position_json.h"
#include "sim/sim_runner.h"
#include "util/assert.h"

#include <boost/json.hpp>

#include <limits>

namespace scribblez {

namespace json = boost::json;

bool read_trajectory_decision(const std::string& gcg_text, const Dictionary& dict, bool open_leaves,
                              TrajectoryDecision* out, std::string* error) {
  ParsedGcgPosition& p = out->position;
  if (!read_gcg_position(gcg_text, open_leaves, &p, error)) return false;
  // The ranking request TrajectoryRunner::run makes: the opponent rack is
  // the known leave under open leaves, else nothing.
  const MoveRequest req{
    p.board, dict, p.rack, p.opp_leave, p.scores[p.mover], p.scores[1 - p.mover], p.bag_size};
  out->legal_moves = equity_top_k(req, std::numeric_limits<int>::max());
  return true;
}

void encode_trajectory_decision(const TrajectoryDecision& d, const InputEncodingSpec& spec,
                                float* out, int* score_diff) {
  const ParsedGcgPosition& p = d.position;
  // apply_move attributes each move to the encoder's own turn order (seat 0
  // first), which the recorded seats follow -- the same replay the generator
  // and the training path run.
  GameStateEncoder enc{spec};
  for (const ParsedGcgTurn& t : p.game.turns) enc.apply_move(t.record.move);
  RELEASE_ASSERT(enc.active_player() == p.mover);
  // The cross-check planes read the board's move-generation caches.
  enc.board().ensure_movegen_caches(*spec.dict);
  if (spec.opp_leave_input) {
    enc.encode_input(p.mover, p.rack, p.opp_leave, out);
  } else {
    enc.encode_input(p.mover, p.rack, out);
  }
  *score_diff = enc.score(p.mover) - enc.score(1 - p.mover);
}

std::string trajectory_decision_board_json(const TrajectoryDecision& d) {
  const ParsedGcgPosition& p = d.position;
  const int opp = 1 - p.mover;
  json::object o =
    position_state_object_pov(p.board, p.rack, p.scores[p.mover], p.scores[opp],
                              p.game.player_names[p.mover], p.game.player_names[opp]);
  o["mover"] = p.mover;
  o["opp_leave"] = p.opp_leave.to_string();
  o["last_move"] =
    p.game.turns.empty() ? json::array{} : move_squares(p.game.turns.back().record.move);
  json::array moves;
  moves.reserve(d.legal_moves.size());
  for (const Move& m : d.legal_moves) moves.emplace_back(move_notation(p.board, m));
  o["moves"] = std::move(moves);
  return json::serialize(o);
}

}  // namespace scribblez
