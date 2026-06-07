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
                          const uint8_t* flips, std::size_t n_indices, int64_t output_row_start,
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
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));

  build_requests(metas, hdr->num_games, local_indices, n_indices);

  std::size_t cursor = 0;
  while (cursor < requests_.size()) {
    const uint32_t game_idx = requests_[cursor].game;
    replay_game(buf, game_idx, flips, output_row_start, output, cursor);
  }
}

void BlockDecoder::build_requests(const GameMetadata* metas, uint32_t num_games,
                                  const int64_t* local_indices, std::size_t n_indices) {
  // Prefix sum: prefix[g] == total positions in games [0, g).
  std::vector<uint32_t> prefix(num_games + 1, 0);
  for (uint32_t g = 0; g < num_games; ++g) {
    prefix[g + 1] = prefix[g] + metas[g].num_positions;
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
                               int64_t output_row_start, float* output, std::size_t& cursor) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];

  const InitialRacks* ir = reinterpret_cast<const InitialRacks*>(buf + gm.start_offset);
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  enc_ = GameStateEncoder{};
  racks_[0] = rack_from_initial(ir->p0, ir->n0);
  racks_[1] = rack_from_initial(ir->p1, ir->n1);

  // Walk turns, emitting samples whose intra index matches.
  uint32_t pos = 0;  // position index within this game
  for (uint32_t k = 0;
       k < gm.num_turns && cursor < requests_.size() && requests_[cursor].game == game_idx; ++k) {
    // Pre-move row for turn k.
    while (cursor < requests_.size() && requests_[cursor].game == game_idx &&
           requests_[cursor].intra == pos) {
      emit_row(requests_[cursor], flips, output_row_start, output,
               /*maybe_play_for_post=*/nullptr, gm, turns);
      ++cursor;
    }
    ++pos;

    // Post-move row only for PLAY turns; emitted from the SAME pre-apply
    // state (encode_input_post_play is non-mutating).
    if (turns[k].move.type == MoveType::PLAY) {
      while (cursor < requests_.size() && requests_[cursor].game == game_idx &&
             requests_[cursor].intra == pos) {
        emit_row(requests_[cursor], flips, output_row_start, output,
                 /*maybe_play_for_post=*/&turns[k].move, gm, turns);
        ++cursor;
      }
      ++pos;
    }

    // Advance both the encoder and the per-game rack tracking.
    const int active = enc_.active_player();
    if (turns[k].move.type == MoveType::PLAY || turns[k].move.type == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[active], turns[k].move);
    }
    for (uint8_t i = 0; i < turns[k].num_drawn; ++i) racks_[active].add(turns[k].drawn[i]);
    enc_.apply_move(turns[k].move);
  }
  // Defensive: skip past any unconsumed requests for this game (shouldn't
  // happen given the prefix sums computed in build_requests).
  while (cursor < requests_.size() && requests_[cursor].game == game_idx) ++cursor;
}

void BlockDecoder::emit_row(const Request& req, const uint8_t* flips, int64_t output_row_start,
                            float* output, const Move* maybe_play_for_post, const GameMetadata& gm,
                            const TurnBlob* turns) {
  const bool flip = flips[req.out_i] != 0;
  float* row = output + (output_row_start + static_cast<int64_t>(req.out_i)) * kRowFloats;

  // The active player POV for label heads is whoever is on move at this
  // sample. For a post-move row that's still the player who just played
  // (no turn alternation has happened in the encoder yet).
  const int active = enc_.active_player();
  const int opp = 1 - active;

  if (maybe_play_for_post != nullptr) {
    enc_.encode_input_post_play(*maybe_play_for_post, racks_[active], racks_[opp].size(), flip,
                                row);
  } else {
    enc_.encode_input(racks_[active], racks_[opp].size(), flip, row);
  }

  // The "opponent next move" comes from the opponent's reply to the active
  // player's turn k. With strict alternation that's turns[k+1] -- the same
  // expression whether this is a pre-move or post-move row (in both cases
  // the encoder has not yet applied turn k).
  const int next_turn_idx = enc_.turn_index() + 1;
  GameLogView view{};
  if (next_turn_idx < static_cast<int>(gm.num_turns)) {
    view.next_move = turns[next_turn_idx].move;
    view.has_next_move = true;
  }
  view.active_player = active;
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
