#include "data/block_decoder.h"

#include "data/binary_log.h"
#include "data/data_loader.h"
#include "serve/position_json.h"
#include "training/max_move_per_lane_task.h"
#include "training/training_task.h"

#include <format>
#include <iostream>

namespace scribblez {
namespace binlog {

namespace {

// Column-letter + 1-based-row coordinate, e.g. (r=7, c=7) -> "H8".
std::string square_name(int r, int c) { return std::format("{}{}", char('A' + c), r + 1); }

// One line, from the move's own stored data alone. A PLAY lists only the tiles
// it PLACED (blanks lowercased) with the anchor square and orientation; the
// board diagram beside it shows the full word.
std::string describe_move(const Move& m) {
  if (m.type() == MoveType::PASS) return "PASS";

  std::string tiles;
  for (int i = 0; i < m.num_glyphs(); ++i) {
    char ch = m.glyph(i).to_char();
    if (m.glyph(i).is_blank()) ch = char(ch - 'A' + 'a');
    tiles.push_back(ch);
  }
  if (m.type() == MoveType::EXCHANGE) return "EXCHANGE " + tiles;

  // PLAY: locate the first placed square from the lane mask (bit k = lane cell k).
  uint16_t mask = m.square_mask();
  int along = 0;
  while (mask && (mask & 1u) == 0) {
    mask >>= 1;
    ++along;
  }
  const bool horizontal = m.horizontal();
  const int r = horizontal ? m.start() : along;
  const int c = horizontal ? along : m.start();
  return "PLAY " + tiles + " @ " + square_name(r, c) + (horizontal ? " across" : " down");
}

}  // namespace

int BlockDecoder::row_floats_for(DecodeTask task, const InputEncodingSpec& spec) {
  if (task == DecodeTask::kMaxMovePerLane) return MaxMovePerLaneTask::kRowFloats;
  return input_floats(spec) + kLabelFloats;
}

GameLog BlockDecoder::game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn) {
  return make_game_view(buf, game_idx, scratch_, sampled_turn);
}

void BlockDecoder::decode(const char* buf, const std::string& path, int64_t local_start,
                          int64_t n_rows, const uint8_t* flips, bool post_move,
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
    const uint32_t game_idx = uint32_t(local_start + i);
    uint32_t sampled = 0;
    const GameLog g = game_view(buf, game_idx, &sampled);
    pos_.encode_row<PositionEvalTask>(g, int(sampled), post_move, flips[i] != 0,
                                      output + (output_row_start + i) * row_floats_);
  }
}

void BlockDecoder::decode_one(const char* buf, const std::string& path, uint32_t game_idx,
                              uint32_t turn_idx, bool flip, bool post_move, int64_t output_row,
                              float* output) {
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
  // The lane task encodes the pre-move position (its labels come from enumerating
  // legal moves at the position), so it ignores the caller's post_move flag.
  if (task_ == DecodeTask::kMaxMovePerLane) {
    pos_.encode_row<MaxMovePerLaneTask>(g, int(turn_idx), /*post_move=*/false, flip, out);
  } else {
    pos_.encode_row<PositionEvalTask>(g, int(turn_idx), post_move, flip, out);
  }
}

void BlockDecoder::encode_score_diff_sweep(const char* buf, uint32_t game_idx, bool post_move,
                                           int diff_lo, int diff_hi, float* out) {
  uint32_t sampled = 0;
  const GameLog g = game_view(buf, game_idx, &sampled);
  pos_.encode_score_diff_sweep(g, int(sampled), post_move, diff_lo, diff_hi, out);
}

std::string BlockDecoder::dump_position(const char* buf, uint32_t game_idx, bool post_move) {
  uint32_t sampled = 0;
  const GameLog g = game_view(buf, game_idx, &sampled);
  const int mover = pos_.replay_to_sampled(g, int(sampled), post_move);
  const int opp = 1 - mover;
  const GameStateEncoder& enc = pos_.enc();
  const int active = enc.score(mover);
  const int other = enc.score(opp);

  std::string s;
  s += std::format("game_idx={} kind={}\n", game_idx, post_move ? "post_move" : "pre_move");
  s += std::format("POV player={}  score: active={} opp={} diff={}\n", mover, active, other,
                   active - other);
  s += std::format("POV rack (leave): {}\n", pos_.rack(mover).to_string());
  s += std::format("last self move: {}\n", describe_move(enc.last_move_by(mover)));
  s += std::format("last opp move:  {}\n", describe_move(enc.last_move_by(opp)));
  s += enc.board().to_string();
  return s;
}

std::string BlockDecoder::dump_position_json(const char* buf, uint32_t game_idx, bool post_move) {
  uint32_t sampled = 0;
  const GameLog g = game_view(buf, game_idx, &sampled);
  const int mover = pos_.replay_to_sampled(g, int(sampled), post_move);
  const int opp = 1 - mover;
  const GameStateEncoder& enc = pos_.enc();
  boost::json::object o = position_state_object_pov(enc.board(), pos_.rack(mover), enc.score(mover),
                                                    enc.score(opp), "You", "Opponent");
  // The squares of the move that produced this (post-move) position, for the
  // renderer to highlight. For a pre-move snapshot this is the POV player's
  // previous play, still the most recent move they made.
  o["last_move"] = move_squares(enc.last_move_by(mover));
  return boost::json::serialize(o);
}

}  // namespace binlog
}  // namespace scribblez
