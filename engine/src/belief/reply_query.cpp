#include "belief/reply_query.h"

#include "encoding/input_encoder.h"
#include "encoding/leave_inference_input_encoder.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "util/assert.h"

#include <algorithm>

namespace scribblez::belief {

namespace {

// Tiles `m` takes from the mover's rack: played, or put back in the bag.
int tiles_off_rack(const Move& m) { return m.type() == MoveType::PASS ? 0 : m.num_glyphs(); }

// The bag after the chooser plays `m` and draws.
int bag_after(int bag, const Move& m) {
  return m.type() == MoveType::PLAY ? bag - std::min(bag, m.num_glyphs()) : bag;
}

LeaveRowCause reply_cause(const Move& candidate, int leave_length) {
  if (candidate.type() == MoveType::PASS) return LeaveRowCause::kPass;
  if (leave_length == 0) return LeaveRowCause::kEmptyLeave;
  return LeaveRowCause::kValid;
}

// The opponent's history: the candidate, its mask left to each query, then the
// chooser's earlier moves under the public mask.
LeaveHistory reply_history(const ReplyPosition& pos, const Move& candidate,
                           const Dictionary& dict) {
  LeaveHistory h;
  h.slots[0] = history_slot(pos.board, dict, candidate, pos.bag, kAnyLeaveKind);
  h.slots[0].forgone = compute_forgone_plays(pos.board, dict, candidate);
  const size_t older = std::min(pos.earlier.size(), size_t(LeaveHistory::kSlots - 1));
  for (size_t i = 0; i < older; ++i) {
    const ChooserMove& e = pos.earlier[i];
    h.slots[i + 1] = history_slot(e.board_before, dict, e.move, e.bag_before, kAnyLeaveKind);
    h.slots[i + 1].forgone = e.forgone;
  }
  return h;
}

Board board_after(const Board& before, const Move& m, const Dictionary& dict) {
  Board after = before;
  if (m.type() == MoveType::PLAY) after.apply(m);
  after.ensure_movegen_caches(dict);
  return after;
}

}  // namespace

ReplyView::ReplyView(const ReplyPosition& pos, const Move& candidate, const Dictionary& dict)
    : dict_(dict),
      after_(board_after(pos.board, candidate, dict)),
      history_(reply_history(pos, candidate, dict)),
      length_(pos.chooser_rack.size() - tiles_off_rack(candidate)),
      cause_(reply_cause(candidate, length_)),
      bag_(bag_after(pos.bag, candidate)),
      score_diff_(pos.opp_score - pos.chooser_score -
                  (candidate.type() == MoveType::PLAY ? candidate.score() : 0)) {}

LeaveQuery ReplyView::query(const Rack& opp_rack) const {
  LeaveQuery q;
  q.length = length_;
  q.cause = cause_;
  q.bag = bag_;
  q.leave_pool = after_.unseen_tiles(opp_rack);
  LeaveHistory history = history_;
  history.slots[0].leave_kinds = leave_kinds_of(q.leave_pool);
  const LeaveInferenceInput in{
    &after_, opp_rack, score_diff_, bag_, length_, &history, LeaveInferenceView::kObserver};
  q.input.resize(LeaveInferenceInputEncoder::kInputFloats);
  LeaveInferenceInputEncoder::encode(in, dict_, q.input.data());
  return q;
}

std::vector<ChooserMove> chooser_moves_from_replay(const GameLog& g, int turn,
                                                   std::span<const ForgonePlays> forgone,
                                                   const Dictionary& dict) {
  binlog::PositionEncoder encoder(InputEncodingSpec{&dict});
  std::vector<ChooserMove> out;
  for (int k = turn - 2; k >= 0 && out.size() < LeaveHistory::kSlots - 1; k -= 2) {
    RELEASE_ASSERT(k < int(forgone.size()), "turn {} has no forgone plays ({} stored)", k,
                   forgone.size());
    encoder.replay_to_sampled(g, k, /*post_move=*/false);
    out.push_back({g.records[k].move, encoder.enc().board(), encoder.bag_size(), forgone[k]});
  }
  return out;
}

}  // namespace scribblez::belief
