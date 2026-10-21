#include "belief/slog_leave_beliefs.h"

#include "data/block_decoder.h"
#include "encoding/input_encoder.h"
#include "lexicon/dictionary.h"

namespace scribblez::belief {

namespace {

using binlog::BlockDecoder;
using binlog::DecodeTask;

// The observer-visible fields of a decoded row's OppLeaveTarget.
void read_target(const float* target, LeaveQuery* q) {
  q->length = int(target[OppLeaveTarget::kLengthOffset]);
  q->cause = LeaveRowCause(int(target[OppLeaveTarget::kCauseOffset]));
  for (int t = 0; t < TILE_KINDS; ++t)
    q->leave_pool.add(Tile::of(t), int(target[OppLeaveTarget::kPoolOffset + t]));
  q->bag = int(target[OppLeaveTarget::kBagOffset]);
}

}  // namespace

SlogLeaveBeliefs::SlogLeaveBeliefs(const std::vector<char>& slog, const std::string& slog_path,
                                   const Dictionary& dict, const LeaveValueArrays& table,
                                   LeaveParamsFn predict)
    : slog_(slog),
      slog_path_(slog_path),
      dict_(dict),
      table_(table),
      predict_(std::move(predict)),
      forgone_(forgone_path_of(slog_path), slog.data(), dict.content_hash()) {}

LeaveQuery SlogLeaveBeliefs::query(const binlog::GamePositionIndex& at) const {
  const InputEncodingSpec spec{&dict_};
  BlockDecoder decoder(spec, DecodeTask::kLeaveInference);
  std::vector<float> row(BlockDecoder::row_floats_for(DecodeTask::kLeaveInference, spec));
  decoder.decode_one(slog_.data(), slog_path_, at.game_idx, at.turn_idx, /*transpose=*/false,
                     /*post_move=*/false, /*output_row=*/0, row.data(), &forgone_);
  LeaveQuery q;
  read_target(row.data() + LeaveInferenceTask::kInputFloats, &q);
  row.resize(LeaveInferenceTask::kInputFloats);
  q.input = std::move(row);
  return q;
}

std::shared_ptr<const LeaveSampler> SlogLeaveBeliefs::belief(
  const binlog::GamePositionIndex& at) const {
  const LeaveQuery q = query(at);
  if (!q.modelled()) return nullptr;
  LeaveParams params;
  predict_(q.input.data(), 1, &params);
  return std::make_shared<const LeaveSampler>(params, q.length, q.leave_pool, table_);
}

}  // namespace scribblez::belief
