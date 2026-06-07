#include "scribblez/block_decoder.h"

#include "scribblez/data_loader.h"
#include "scribblez/input_encoder.h"
#include "scribblez/label_encoder.h"

#include <algorithm>
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
    [[maybe_unused]] bool ok = rack.remove(m.glyphs[i].rack_tile());
    assert(ok);
  }
}

// Reconstruct a Rack from the on-disk InitialRacks slot for one player.
Rack rack_from_initial(const std::array<Tile, RACK_SIZE>& tiles, uint8_t n) {
  Rack r;
  for (uint8_t k = 0; k < n; ++k) r.add(tiles[k]);
  return r;
}

}  // namespace

void BlockDecoder::decode(const char* buf, const std::string& path, const int64_t* local_indices,
                          const uint8_t* flips, std::size_t n_indices, bool post_move,
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
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));

  build_requests(metas, hdr->num_games, local_indices, n_indices);

  std::size_t cursor = 0;
  while (cursor < requests_.size()) {
    const uint32_t game_idx = requests_[cursor].game;
    replay_game(buf, game_idx, flips, post_move, output_row_start, output, cursor);
  }
}

void BlockDecoder::build_requests(const GameMetadata* metas, uint32_t num_games,
                                  const int64_t* local_indices, std::size_t n_indices) {
  // Prefix sum: prefix[g] == total positions in games [0, g) == sum of
  // num_turns over those games (the master-array universe is one row per
  // turn, shared by both pre-move and post-move sampling).
  std::vector<uint32_t> prefix(num_games + 1, 0);
  for (uint32_t g = 0; g < num_games; ++g) {
    prefix[g + 1] = prefix[g] + metas[g].num_turns;
  }

  requests_.clear();
  requests_.resize(n_indices);
  uint32_t game_cursor = 0;
  for (std::size_t i = 0; i < n_indices; ++i) {
    const int64_t li = local_indices[i];
    while (game_cursor + 1 < num_games && prefix[game_cursor + 1] <= li) ++game_cursor;
    requests_[i].game = game_cursor;
    requests_[i].intra = static_cast<uint32_t>(static_cast<uint64_t>(li) - prefix[game_cursor]);
    requests_[i].out_i = static_cast<uint32_t>(i);
  }
  std::sort(requests_.begin(), requests_.end(), [](const Request& a, const Request& b) {
    if (a.game != b.game) return a.game < b.game;
    return a.intra < b.intra;
  });
}

void BlockDecoder::replay_game(const char* buf, uint32_t game_idx, const uint8_t* flips,
                               bool post_move, int64_t output_row_start, float* output,
                               std::size_t& cursor) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];

  const InitialRacks* ir = reinterpret_cast<const InitialRacks*>(buf + gm.start_offset);
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  enc_ = GameStateEncoder{};
  racks_[0] = rack_from_initial(ir->p0, ir->n0);
  racks_[1] = rack_from_initial(ir->p1, ir->n1);

  // Walk turns; one sample per turn. requests_[cursor].intra is just the
  // turn index k. For pre-move sampling we emit BEFORE applying the move;
  // for post-move sampling we emit AFTER applying the move and removing
  // played/exchanged tiles from the mover's rack but BEFORE the mover
  // draws replacements.
  for (uint32_t k = 0;
       k < gm.num_turns && cursor < requests_.size() && requests_[cursor].game == game_idx; ++k) {
    const int mover = enc_.active_player();

    if (!post_move) {
      while (cursor < requests_.size() && requests_[cursor].game == game_idx &&
             requests_[cursor].intra == k) {
        emit_row(requests_[cursor], flips, output_row_start, output, mover, gm, turns);
        ++cursor;
      }
    }

    // Apply the move's board/score/rack effect from the mover's side.
    if (turns[k].move.type == MoveType::PLAY || turns[k].move.type == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[mover], turns[k].move);
    }
    enc_.apply_move(turns[k].move);

    if (post_move) {
      while (cursor < requests_.size() && requests_[cursor].game == game_idx &&
             requests_[cursor].intra == k) {
        emit_row(requests_[cursor], flips, output_row_start, output, mover, gm, turns);
        ++cursor;
      }
    }

    // Mover draws replacements; this finishes their turn from a bookkeeping
    // POV. We update racks_ *after* the post-row emit so the post-row sees
    // the pre-draw rack (matching the unseen-pool composition the active
    // player observes immediately after placing their tiles).
    for (uint8_t i = 0; i < turns[k].num_drawn; ++i) racks_[mover].add(turns[k].drawn[i]);
  }
  // Defensive: skip past any unconsumed requests for this game (shouldn't
  // happen given the prefix sums computed in build_requests).
  while (cursor < requests_.size() && requests_[cursor].game == game_idx) ++cursor;
}

void BlockDecoder::emit_row(const Request& req, const uint8_t* flips, int64_t output_row_start,
                            float* output, int pov_player, const GameMetadata& gm,
                            const TurnBlob* turns) {
  const bool flip = flips[req.out_i] != 0;
  float* row = output + (output_row_start + static_cast<int64_t>(req.out_i)) * kRowFloats;

  enc_.encode_input(pov_player, racks_[pov_player], flip, row);

  // The "opponent next move" is the opponent's reply to pov_player's most
  // recent turn. Cases:
  //   pre-move sample  -> enc_.active_player() == pov_player (turn k not
  //                       applied yet), so the reply is turns[k+1] =
  //                       turns[enc_.turn_index() + 1].
  //   post-move sample -> enc_.active_player() == 1 - pov_player (turn k
  //                       already applied), so the reply is turns[k+1] =
  //                       turns[enc_.turn_index()].
  const int next_idx = enc_.turn_index() + (enc_.active_player() == pov_player ? 1 : 0);

  GameLogView view{};
  if (next_idx < static_cast<int>(gm.num_turns)) {
    view.next_move = turns[next_idx].move;
    view.has_next_move = true;
  }
  view.active_player = pov_player;
  view.final_score_p0 = gm.final_score_p0;
  view.final_score_p1 = gm.final_score_p1;
  view.apply_flip = flip;

  float* heads[kNumLabelHeads] = {
    row + kInputFloats,
    row + kInputFloats + kWldFloats,
    row + kInputFloats + kWldFloats + kScoreDiffFloats,
  };
  encode_labels(view, heads);
}

}  // namespace binlog
}  // namespace scribblez
