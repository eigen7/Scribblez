#include "scribblez/block_decoder.h"

#include "scribblez/binary_log.h"
#include "scribblez/data_loader.h"
#include "scribblez/input_encoder.h"
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

void BlockDecoder::replay_and_emit(const char* buf, uint32_t game_idx, bool flip, bool post_move,
                                   int64_t out_row, float* output) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];

  const InitialRacks* ir = reinterpret_cast<const InitialRacks*>(buf + gm.start_offset);
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  enc_ = GameStateEncoder{};
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

  float* row = output + out_row * kRowFloats;
  enc_.encode_input(mover, racks_[mover], flip, row);

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
