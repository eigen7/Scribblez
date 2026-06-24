#include "scribblez/block_decoder.h"

#include "scribblez/binary_log.h"
#include "scribblez/data_loader.h"
#include "scribblez/position_json.h"

#include <iostream>

namespace scribblez {
namespace binlog {

namespace {

// Column-letter + 1-based-row coordinate, e.g. (r=7, c=7) -> "H8".
std::string square_name(int r, int c) {
  std::string s;
  s.push_back(static_cast<char>('A' + c));
  s += std::to_string(r + 1);
  return s;
}

// Board-independent one-line description of a move, built from the move's own
// stored data. The PLAY string lists only the tiles this move PLACED (blanks
// lowercased), with the anchor square and orientation; connecting tiles already
// on the board are not part of it -- the board diagram shows the full word.
std::string describe_move(const Move& m) {
  if (m.type() == MoveType::PASS) return "PASS";

  std::string tiles;
  for (int i = 0; i < m.num_glyphs(); ++i) {
    char ch = m.glyph(i).to_char();
    if (m.glyph(i).is_blank()) ch = static_cast<char>(ch - 'A' + 'a');
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

// The [row, col] board squares a PLAY placed tiles on (empty for EXCHANGE/PASS),
// derived from the move's lane mask -- mirrors encode_placement_plane.
boost::json::array move_squares(const Move& m) {
  boost::json::array squares;
  if (m.type() != MoveType::PLAY) return squares;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    squares.emplace_back(boost::json::array{r, c});
  }
  return squares;
}

}  // namespace

GameLog BlockDecoder::game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];
  const InitialRacks* ir = reinterpret_cast<const InitialRacks*>(buf + gm.start_offset);
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  scratch_.resize(gm.num_turns);
  for (uint32_t k = 0; k < gm.num_turns; ++k) {
    scratch_[k].move = turns[k].move;
    scratch_[k].drawn = turns[k].drawn;
  }

  GameLog g;
  g.initial_racks[0] = ir->p0;
  g.initial_racks[1] = ir->p1;
  g.initial_scores = {gm.initial_score_p0, gm.initial_score_p1};
  g.final_scores = {gm.final_score_p0, gm.final_score_p1};
  g.records = scratch_.data();
  g.num_records = static_cast<int>(gm.num_turns);
  if (sampled_turn) *sampled_turn = gm.sampled_turn;
  return g;
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
    const uint32_t game_idx = static_cast<uint32_t>(local_start + i);
    uint32_t sampled = 0;
    const GameLog g = game_view(buf, game_idx, &sampled);
    pos_.encode_row(g, static_cast<int>(sampled), post_move, flips[i] != 0,
                    output + (output_row_start + i) * kRowFloats);
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
  pos_.encode_row(g, static_cast<int>(turn_idx), post_move, flip, output + output_row * kRowFloats);
}

void BlockDecoder::encode_score_diff_sweep(const char* buf, uint32_t game_idx, bool post_move,
                                           int diff_lo, int diff_hi, float* out) {
  uint32_t sampled = 0;
  const GameLog g = game_view(buf, game_idx, &sampled);
  pos_.encode_score_diff_sweep(g, static_cast<int>(sampled), post_move, diff_lo, diff_hi, out);
}

std::string BlockDecoder::dump_position(const char* buf, uint32_t game_idx, bool post_move) {
  uint32_t sampled = 0;
  const GameLog g = game_view(buf, game_idx, &sampled);
  const int mover = pos_.replay_to_sampled(g, static_cast<int>(sampled), post_move);
  const int opp = 1 - mover;
  const GameStateEncoder& enc = pos_.enc();
  const int active = enc.score(mover);
  const int other = enc.score(opp);

  std::string s;
  s += "game_idx=" + std::to_string(game_idx) + " kind=" + (post_move ? "post_move" : "pre_move") +
       "\n";
  s += "POV player=" + std::to_string(mover) + "  score: active=" + std::to_string(active) +
       " opp=" + std::to_string(other) + " diff=" + std::to_string(active - other) + "\n";
  s += "POV rack (leave): " + pos_.rack(mover).to_string() + "\n";
  s += "last self move: " + describe_move(enc.last_move_by(mover)) + "\n";
  s += "last opp move:  " + describe_move(enc.last_move_by(opp)) + "\n";
  s += enc.board().to_string();
  return s;
}

std::string BlockDecoder::dump_position_json(const char* buf, uint32_t game_idx, bool post_move) {
  uint32_t sampled = 0;
  const GameLog g = game_view(buf, game_idx, &sampled);
  const int mover = pos_.replay_to_sampled(g, static_cast<int>(sampled), post_move);
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
