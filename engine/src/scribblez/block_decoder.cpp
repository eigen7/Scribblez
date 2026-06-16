#include "scribblez/block_decoder.h"

#include "scribblez/binary_log.h"
#include "scribblez/data_loader.h"
#include "scribblez/input_encoder.h"
#include "scribblez/position_json.h"
#include "scribblez/training_targets.h"

#include <cassert>
#include <iostream>

namespace scribblez {
namespace binlog {

namespace {

// Replace all glyphs the move places with empty tiles in `rack` (PLAY: the
// tiles being placed; EXCHANGE: the tiles being swapped out).
void remove_played_or_exchanged(Rack& rack, const Move& m) {
  const int n = m.num_glyphs();
  for (int i = 0; i < n; ++i) {
    [[maybe_unused]] bool ok = rack.remove(m.glyph(i).rack_tile());
    assert(ok);
  }
}

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
// (Reconstructing the full word via Move::main_word needs the pre-placement
// board, which a position dump does not have.)
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
    const bool flip = flips[i] != 0;
    replay_and_emit(buf, game_idx, flip, post_move, output_row_start + i, output);
  }
}

int BlockDecoder::replay_to_sampled(const char* buf, uint32_t game_idx, bool post_move) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];

  const InitialRacks* ir = reinterpret_cast<const InitialRacks*>(buf + gm.start_offset);
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  enc_ = GameStateEncoder{std::array<int, 2>{gm.initial_score_p0, gm.initial_score_p1}};
  racks_[0] = ir->p0;
  racks_[1] = ir->p1;

  // Replay turns [0, sampled_turn) silently (apply move + remove played
  // tiles + add drawn tiles). The mover of turn k alternates from active=0.
  const uint32_t target = gm.sampled_turn;
  for (uint32_t k = 0; k < target; ++k) {
    const int mover = enc_.active_player();
    if (turns[k].move.type() == MoveType::PLAY || turns[k].move.type() == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[mover], turns[k].move);
    }
    enc_.apply_move(turns[k].move);
    for (Tile t : turns[k].drawn.tiles()) {
      if (t.is_empty()) break;
      racks_[mover].add(t);
    }
  }

  // At this point enc_ holds the pre-move state for turn `target`.
  const int mover = enc_.active_player();
  if (post_move) {
    if (turns[target].move.type() == MoveType::PLAY ||
        turns[target].move.type() == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[mover], turns[target].move);
    }
    enc_.apply_move(turns[target].move);
    // racks_[mover] is now the pre-draw rack; do NOT add turns[target].drawn.
  }
  return mover;
}

void BlockDecoder::encode_score_diff_sweep(const char* buf, uint32_t game_idx, bool post_move,
                                           int diff_lo, int diff_hi, float* out) {
  const int mover = replay_to_sampled(buf, game_idx, post_move);
  int64_t i = 0;
  for (int d = diff_lo; d <= diff_hi; ++d, ++i) {
    enc_.encode_input_with_score_diff(mover, racks_[mover], d, /*apply_flip=*/false,
                                      out + i * kInputFloats);
  }
}

std::string BlockDecoder::dump_position(const char* buf, uint32_t game_idx, bool post_move) {
  const int mover = replay_to_sampled(buf, game_idx, post_move);
  const int opp = 1 - mover;
  const Board& board = enc_.board();
  const int active = enc_.score(mover);
  const int other = enc_.score(opp);

  std::string s;
  s += "game_idx=" + std::to_string(game_idx) + " kind=" + (post_move ? "post_move" : "pre_move") +
       "\n";
  s += "POV player=" + std::to_string(mover) + "  score: active=" + std::to_string(active) +
       " opp=" + std::to_string(other) + " diff=" + std::to_string(active - other) + "\n";
  s += "POV rack (leave): " + racks_[mover].to_string() + "\n";
  s += "last self move: " + describe_move(enc_.last_move_by(mover)) + "\n";
  s += "last opp move:  " + describe_move(enc_.last_move_by(opp)) + "\n";
  s += board.to_string();
  return s;
}

std::string BlockDecoder::dump_position_json(const char* buf, uint32_t game_idx, bool post_move) {
  const int mover = replay_to_sampled(buf, game_idx, post_move);
  const int opp = 1 - mover;
  boost::json::object o = position_state_object_pov(enc_.board(), racks_[mover], enc_.score(mover),
                                                    enc_.score(opp), "You", "Opponent");
  // The squares of the move that produced this (post-move) position, for the
  // renderer to highlight. For a pre-move snapshot this is the POV player's
  // previous play, still the most recent move they made.
  o["last_move"] = move_squares(enc_.last_move_by(mover));
  return boost::json::serialize(o);
}

void BlockDecoder::replay_and_emit(const char* buf, uint32_t game_idx, bool flip, bool post_move,
                                   int64_t out_row, float* output) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  const int mover = replay_to_sampled(buf, game_idx, post_move);

  float* row = output + out_row * kRowFloats;
  enc_.encode_input(mover, racks_[mover], flip, row);

  const uint32_t target = gm.sampled_turn;

  // The "opponent next move" -- whichever upcoming turn was played by
  // (1 - mover). Pre-move: that's turn target+1. Post-move: enc_ has
  // advanced past `target` (turn_index == target+1), so it's turn
  // target+1 as well (enc_.active_player() is now opp; opp's next move
  // is at turn target+1).
  const uint32_t next_idx = target + 1;
  GameLogView view{};
  if (next_idx < gm.num_turns) {
    view.next_move = turns[next_idx].move;
    view.has_next_move = true;
  }
  view.active_player = mover;
  view.final_score_p0 = gm.final_score_p0;
  view.final_score_p1 = gm.final_score_p1;
  view.apply_flip = flip;

  AllTargets::encode_all(view, row + kInputFloats);
}

}  // namespace binlog
}  // namespace scribblez
