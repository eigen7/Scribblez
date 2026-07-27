#include "endgame/move_list_memo.h"

#include "game/board.h"
#include "game/movegen.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "util/math.h"

namespace scribblez {

const std::vector<Move>& MoveListMemo::generate(const Board& board, const Dictionary& dict,
                                                const Rack& rack, uint64_t board_hash) {
  if (!enabled_) {
    scratch_ = MoveGenerator(board, dict).generate(rack);
    return scratch_;
  }
  const uint64_t key = board_hash ^ util::splitmix64(rack.bits());
  const auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;
  if (cache_.size() >= kCap) cache_.clear();
  return cache_.emplace(key, MoveGenerator(board, dict).generate(rack)).first->second;
}

}  // namespace scribblez
