#include "data/block_decoder.h"

#include "data/binary_log.h"
#include "data/data_loader.h"
#include "data/forgone_play_log.h"
#include "training/leaf_eval_task.h"
#include "training/leave_inference_task.h"
#include "training/max_move_per_lane_task.h"
#include "training/training_task.h"
#include "util/assert.h"
#include "util/exception.h"

#include <cstdlib>
#include <iostream>

namespace scribblez {
namespace binlog {

namespace {

int eligible_end(const char* buf, uint32_t game_idx) {
  return reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader))[game_idx].eligible_end;
}

}  // namespace

int BlockDecoder::input_floats_for(DecodeTask task, const InputEncodingSpec& spec) {
  switch (task) {
    case DecodeTask::kPositionEval:
      return input_floats(spec);
    case DecodeTask::kMaxMovePerLane:
      return MaxMovePerLaneTask::kInputFloats;
    case DecodeTask::kLeaveInference:
    case DecodeTask::kPublicLeaveInference:
      return LeaveInferenceTask::kInputFloats;
    case DecodeTask::kLeafEval:
      return LeafEvalTask::kInputFloats;
  }
  std::abort();  // unreachable: the switch covers every DecodeTask
}

int BlockDecoder::label_floats_for(DecodeTask task) {
  switch (task) {
    case DecodeTask::kPositionEval:
      return kLabelFloats;
    case DecodeTask::kMaxMovePerLane:
      return MaxMovePerLaneTask::kLabelFloats;
    case DecodeTask::kLeaveInference:
    case DecodeTask::kPublicLeaveInference:
      return LeaveInferenceTask::kLabelFloats;
    case DecodeTask::kLeafEval:
      return LeafEvalTask::kLabelFloats;
  }
  std::abort();  // unreachable: the switch covers every DecodeTask
}

bool BlockDecoder::reads_forgone(DecodeTask task) {
  return task == DecodeTask::kLeaveInference || task == DecodeTask::kPublicLeaveInference;
}

GameLog BlockDecoder::game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn) {
  return make_game_view(buf, game_idx, scratch_, sampled_turn);
}

void BlockDecoder::decode(const char* buf, const std::string& path, int64_t local_start,
                          int64_t n_rows, const uint8_t* transposes, bool post_move,
                          int64_t output_row_start, float* output) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf);
  if (hdr->magic != kMagic) {
    std::cerr << "BlockDecoder: bad magic in " << path << "\n";
    return;
  }
  if (hdr->version != kVersion) {
    std::cerr << "BlockDecoder: version mismatch in " << path << " (file=" << hdr->version
              << " code=" << kVersion << ")\n";
    return;
  }

  for (int64_t i = 0; i < n_rows; ++i) {
    const uint32_t game_idx = local_start + i;
    uint32_t sampled = 0;
    const GameLog g = game_view(buf, game_idx, &sampled);
    pos_.encode_row<PositionEvalTask>(g, int(sampled), post_move, transposes[i] != 0,
                                      output + (output_row_start + i) * row_floats_);
  }
}

void BlockDecoder::decode_one(const char* buf, const std::string& path, uint32_t game_idx,
                              uint32_t turn_idx, bool transpose, bool post_move, int64_t output_row,
                              float* output, const ForgoneSidecar* forgone) {
  RELEASE_ASSERT((forgone != nullptr) == reads_forgone(task_));
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf);
  if (hdr->magic != kMagic) {
    std::cerr << "BlockDecoder: bad magic in " << path << "\n";
    return;
  }
  if (hdr->version != kVersion) {
    std::cerr << "BlockDecoder: version mismatch in " << path << " (file=" << hdr->version
              << " code=" << kVersion << ")\n";
    return;
  }
  const GameLog g = game_view(buf, game_idx, nullptr);
  float* out = output + output_row * row_floats_;
  // The lane and leave inference tasks encode the pre-move position (the lane
  // labels come from enumerating legal moves there, and the leave is inferred
  // before the observer moves), and the leaf eval task the post-move one (a
  // rollout leaf follows a move), so they ignore the caller's post_move flag.
  switch (task_) {
    case DecodeTask::kPositionEval:
      pos_.encode_row<PositionEvalTask>(g, int(turn_idx), post_move, transpose, out);
      return;
    case DecodeTask::kMaxMovePerLane:
      pos_.encode_row<MaxMovePerLaneTask>(g, int(turn_idx), /*post_move=*/false, transpose, out);
      return;
    case DecodeTask::kLeaveInference:
      decode_leave_row<LeaveInferenceTask>(buf, path, g, game_idx, turn_idx, transpose, out,
                                           *forgone);
      return;
    case DecodeTask::kPublicLeaveInference:
      decode_leave_row<PublicLeaveInferenceTask>(buf, path, g, game_idx, turn_idx, transpose, out,
                                                 *forgone);
      return;
    case DecodeTask::kLeafEval:
      pos_.encode_row<LeafEvalTask>(g, int(turn_idx), /*post_move=*/true, transpose, out);
      return;
  }
}

template <typename Task>
void BlockDecoder::decode_leave_row(const char* buf, const std::string& path, const GameLog& g,
                                    uint32_t game_idx, uint32_t turn_idx, bool transpose,
                                    float* out, const ForgoneSidecar& forgone) {
  if (int(turn_idx) >= eligible_end(buf, game_idx)) {
    throw util::Exception("{}: game {} turn {} is past the eligible region", path, game_idx,
                          turn_idx);
  }
  pos_.encode_row<Task>(g, int(turn_idx), /*post_move=*/false, transpose, out,
                        forgone.game(game_idx));
}

const Board& BlockDecoder::replay_board(const char* buf, uint32_t game_idx, uint32_t turn_idx) {
  const GameLog g = game_view(buf, game_idx, nullptr);
  pos_.replay_to_sampled(g, int(turn_idx), /*post_move=*/false);
  return pos_.enc().board();
}

}  // namespace binlog
}  // namespace scribblez
