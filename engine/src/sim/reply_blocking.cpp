#include "sim/reply_blocking.h"

#include "agent/agent.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "util/assert.h"

#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace scribblez {
namespace {

// A play's placement, its tiles on their squares: the move without its score,
// which another candidate's tiles may change, and without its direction, which
// for a single tile they may change too (movegen files a single tile under the
// horizontal pass once it forms a word along both axes).
std::string placement_key(const Move& m) {
  std::string key;
  int i = 0;
  visit_placed_squares(m, [&](int r, int c) {
    key.push_back(char(r * BOARD_SIZE + c));
    key.push_back(char(m.glyph(i++).code()));
  });
  return key;
}

// Each record's first turn, the opponent's reply, or null for a record with
// no turns.
std::vector<const binlog::TurnBlob*> replies(const ProbeReader::Position& pos, size_t records) {
  std::vector<const binlog::TurnBlob*> out(records, nullptr);
  const binlog::TurnBlob* t = pos.turns;
  for (size_t r = 0; r < records; ++r) {
    if (pos.records[r].num_turns > 0) out[r] = t;
    t += pos.records[r].num_turns;
  }
  return out;
}

std::unordered_set<std::string> legal_placements(const Board& board, const Dictionary& dict,
                                                 const Rack& rack) {
  const Rack none;
  const MoveRequest req{board, dict, rack, none, 0, 0, /*bag_size=*/1};
  std::unordered_set<std::string> out;
  for (const Move& m : generate_legal_plays(req)) out.insert(placement_key(m));
  return out;
}

}  // namespace

void reply_blocking(const Dictionary& dict, const Board& root, const ProbeReader::Position& pos,
                    int probes, int stride, uint8_t* out) {
  const int k = int(pos.header->num_candidates);
  RELEASE_ASSERT(stride >= k);
  const size_t records = size_t(k) * size_t(probes);
  const std::vector<const binlog::TurnBlob*> reply = replies(pos, records);
  std::memset(out, 0, records * size_t(stride));
  for (int b = 0; b < k; ++b) {
    Board board = root;
    const Move& candidate = pos.candidates[b].move;
    if (candidate.type() == MoveType::PLAY) board.apply(candidate);
    board.ensure_movegen_caches(dict);
    for (int i = 0; i < probes; ++i) {
      const Rack& rack = pos.records[size_t(b) * probes + i].opp_rack;
      const std::unordered_set<std::string> legal = legal_placements(board, dict, rack);
      for (int a = 0; a < k; ++a) {
        const size_t r = size_t(a) * probes + i;
        if (a == b || reply[r] == nullptr || reply[r]->move.type() != MoveType::PLAY) continue;
        // The common random numbers deal probe i's opponent one rack for all.
        RELEASE_ASSERT(pos.records[r].opp_rack == rack);
        out[r * size_t(stride) + size_t(b)] = !legal.contains(placement_key(reply[r]->move));
      }
    }
  }
}

}  // namespace scribblez
