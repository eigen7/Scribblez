#include "belief/leave_history_recorder.h"

#include "agent/agent.h"
#include "encoding/leave_inference_input_encoder.h"
#include "lexicon/dictionary.h"
#include "util/assert.h"

#include <algorithm>

namespace scribblez::belief {

namespace {

int tiles_off_rack(const Move& m) { return m.type() == MoveType::PASS ? 0 : m.num_glyphs(); }

Rack returned_tiles(const Move& m) {
  Rack out;
  if (m.type() != MoveType::EXCHANGE) return out;
  for (int i = 0; i < m.num_glyphs(); ++i) out.add(m.glyph(i).rack_tile());
  return out;
}

LeaveRowCause own_cause(const Move* last, int leave_length) {
  if (last == nullptr) return LeaveRowCause::kNoOpponentMove;
  if (last->type() == MoveType::PASS) return LeaveRowCause::kPass;
  if (leave_length == 0) return LeaveRowCause::kEmptyLeave;
  return LeaveRowCause::kValid;
}

}  // namespace

void LeaveHistoryRecorder::begin_game() {
  seen_.clear();
  own_racks_.clear();
  board_ = Board{};
  rack_sizes_ = {RACK_SIZE, RACK_SIZE};
  bag_ = board_.unseen_count(2 * RACK_SIZE);
}

void LeaveHistoryRecorder::observe(const Move& move) {
  const int mover = int(seen_.size() % 2);
  seen_.push_back({move, board_, bag_, rack_sizes_[mover], std::nullopt});
  if (move.type() != MoveType::PLAY) return;
  board_.apply(move);
  const int drawn = std::min(bag_, move.num_glyphs());
  rack_sizes_[mover] += drawn - move.num_glyphs();
  bag_ -= drawn;
}

const ForgonePlays& LeaveHistoryRecorder::forgone(int k, const Dictionary& dict) {
  Seen& s = seen_[size_t(k)];
  if (!s.forgone) s.forgone = compute_forgone_plays(s.before, dict, s.move);
  return *s.forgone;
}

Rack LeaveHistoryRecorder::own_exchange(int k) const {
  if (k < 0) return Rack{};
  return returned_tiles(seen_[size_t(k)].move);
}

LeaveHistorySlot LeaveHistoryRecorder::opponent_slot(int k, const Dictionary& dict) {
  const Seen& s = seen_[size_t(k)];
  const auto rack = own_racks_.find(k + 1);
  RELEASE_ASSERT(rack != own_racks_.end(), "no rack recorded at turn {}", k + 1);
  TileCounts unseen = s.before.unseen_tiles(rack->second);
  if (s.move.type() == MoveType::PLAY) {
    for (int i = 0; i < s.move.num_glyphs(); ++i) unseen.remove(s.move.glyph(i).rack_tile());
  }
  LeaveHistorySlot slot = history_slot(s.before, dict, s.move, s.bag_before,
                                       leave_kinds_of(leave_pool(unseen, own_exchange(k - 1))));
  slot.forgone = forgone(k, dict);
  return slot;
}

LeaveHistory LeaveHistoryRecorder::history(int turn, const Dictionary& dict) {
  LeaveHistory h;
  for (int s = 0; s < LeaveHistory::kSlots && turn - 1 - 2 * s >= 0; ++s)
    h.slots[s] = opponent_slot(turn - 1 - 2 * s, dict);
  for (int e = 0; e < LeaveHistory::kObserverExchanges; ++e)
    h.observer_exchanges[e] = own_exchange(turn - 2 - 2 * e);
  return h;
}

LeaveQuery LeaveHistoryRecorder::own_query(const MoveRequest& req) {
  const int turn = int(seen_.size());
  own_racks_[turn] = req.my_rack;
  const LeaveHistory h = history(turn, req.dict);
  const Move* last = turn > 0 ? &seen_.back().move : nullptr;
  LeaveQuery q;
  q.length = last ? seen_.back().mover_rack_before - tiles_off_rack(*last) : 0;
  q.cause = own_cause(last, q.length);
  q.bag = req.bag_size;
  q.leave_pool = leave_pool(req.board.unseen_tiles(req.my_rack), h.observer_exchanges[0]);
  const LeaveInferenceInput in{&req.board, req.my_rack, req.my_score - req.opp_score, req.bag_size,
                               q.length,   &h,          LeaveInferenceView::kObserver};
  q.input.resize(LeaveInferenceInputEncoder::kInputFloats);
  LeaveInferenceInputEncoder::encode(in, req.dict, q.input.data());
  return q;
}

std::vector<ChooserMove> LeaveHistoryRecorder::own_moves(const Dictionary& dict) {
  return moves_from(int(seen_.size()) - 2, dict);
}

std::vector<ChooserMove> LeaveHistoryRecorder::opponent_moves(const Dictionary& dict) {
  return moves_from(int(seen_.size()) - 1, dict);
}

std::vector<ChooserMove> LeaveHistoryRecorder::moves_from(int first, const Dictionary& dict) {
  std::vector<ChooserMove> out;
  for (int k = first; k >= 0 && out.size() < LeaveHistory::kSlots - 1; k -= 2) {
    const Seen& s = seen_[size_t(k)];
    out.push_back({s.move, s.before, s.bag_before, forgone(k, dict)});
  }
  return out;
}

}  // namespace scribblez::belief
