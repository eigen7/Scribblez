#include "training/policy_set_inputs.h"

#include "data/block_decoder.h"
#include "lexicon/hasty_equity.h"
#include "sim/setup_plays.h"
#include "util/assert.h"

#include <algorithm>
#include <limits>

namespace scribblez::policy_set {

namespace {

// Whether the pool keeps `m`, given the greedy move's score. places_blank
// also finds the blank among an exchange's surrendered tiles.
bool in_pool(const Move& m, int greedy_score, int blank_prune_points) {
  const int score = m.type() == MoveType::PLAY ? m.score() : 0;
  return !places_blank(m) || score > greedy_score - blank_prune_points;
}

// Each equity less the best of them, in kValueScale points, floored at
// kEquityFloor. In float, as the inputs have always been computed.
void write_equity_inputs(const std::vector<double>& equities, std::vector<float>* out) {
  out->resize(equities.size());
  float best = -std::numeric_limits<float>::infinity();
  for (double e : equities) best = std::max(best, float(e));
  for (size_t i = 0; i < equities.size(); ++i)
    (*out)[i] = std::max((float(equities[i]) - best) / kValueScale, kEquityFloor);
}

}  // namespace

PoolCandidates pool_candidates(const std::vector<Move>& ranked, int blank_prune_points) {
  int greedy_score = 0;
  for (const Move& m : ranked)
    if (m.type() == MoveType::PLAY) greedy_score = std::max(greedy_score, int(m.score()));
  PoolCandidates out;
  for (size_t i = 0; i < ranked.size(); ++i) {
    if (!in_pool(ranked[i], greedy_score, blank_prune_points)) continue;
    out.moves.push_back(ranked[i]);
    out.equity_ranks.push_back(int32_t(i));
  }
  // The ranking holds a pass only when nothing else is legal.
  if (std::find(out.moves.begin(), out.moves.end(), Move::pass()) == out.moves.end()) {
    out.moves.push_back(Move::pass());
    out.equity_ranks.push_back(-1);
  }
  return out;
}

belief::OppRackSummary opp_rack_summary(const Board& board, const Rack& rack,
                                        const belief::LeaveSampler* leave) {
  return belief::summarize_opp_rack(board.unseen_tiles(rack), leave);
}

std::array<float, kOppRackFloats> encode_opp_rack(const belief::OppRackSummary& summary) {
  std::array<float, kOppRackFloats> out{};
  std::copy(summary.presence.begin(), summary.presence.end(), out.begin());
  out[TILE_KINDS] = summary.modelled ? 1.0f : 0.0f;
  // Divided in double, as the inputs have always been computed.
  out[TILE_KINDS + 1] = float(double(summary.leave_value_mean) / kValueScale);
  out[TILE_KINDS + 2] = float(double(summary.leave_value_sd) / kValueScale);
  return out;
}

void CandidateInputs::encode(const GameStateEncoder& position, const Dictionary& dict,
                             const Rack& rack, int bag_size, const std::vector<Move>& moves) {
  RELEASE_ASSERT(!moves.empty(), "a policy-set position has at least one candidate");
  const int mover = position.active_player();
  features.encode(position.board(), dict, moves.data(), int(moves.size()),
                  position.score(mover) - position.score(1 - mover));
  write_equity_inputs(
    HastyEquity::instance().equities(moves, position.board(), bag_size, Rack{}, rack), &equity);
}

void replay_candidate_inputs(binlog::BlockDecoder& decoder, const Dictionary& dict, const char* buf,
                             const std::string& path, uint32_t game_idx, uint32_t turn_idx,
                             bool transpose, const std::vector<Move>& moves, float* rows,
                             int64_t output_row, CandidateInputs* out) {
  decoder.decode_one(buf, path, game_idx, turn_idx, {.transpose = transpose},
                     /*post_move=*/false, output_row, rows);
  const binlog::PositionEncoder& pos = decoder.position();
  out->encode(pos.enc(), dict, pos.rack(pos.enc().active_player()), pos.bag_size(), moves);
}

}  // namespace scribblez::policy_set
