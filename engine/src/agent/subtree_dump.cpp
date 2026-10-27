#include "agent/subtree_dump.h"

#include "sim/hasty_rollout_task.h"

#include <boost/json.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <mutex>

namespace scribblez {

namespace json = boost::json;

namespace {

constexpr size_t kTopReplies = 3;

size_t argmax(const std::vector<double>& v) {
  return size_t(std::max_element(v.begin(), v.end()) - v.begin());
}

// Bob's most frequent replies over the racks, most frequent first.
json::array top_replies(const std::vector<BobReply>& replies, const Board& board) {
  std::map<std::string, std::pair<int, int>> seen;  // move -> (count, equity rank)
  for (const BobReply& r : replies) {
    auto& [count, rank] = seen[describe_move(r.move, board)];
    ++count;
    rank = r.equity_rank;
  }
  std::vector<std::pair<std::string, std::pair<int, int>>> sorted(seen.begin(), seen.end());
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
  json::array out;
  for (size_t k = 0; k < std::min(kTopReplies, sorted.size()); ++k) {
    out.push_back({{"move", sorted[k].first},
                   {"count", sorted[k].second.first},
                   {"rank", sorted[k].second.second}});
  }
  return out;
}

// What Bob's replies to one of Alice's candidates looked like.
json::object reply_summary(const std::vector<BobReply>& replies, const std::vector<BobReply>& rerun,
                           const Board& board) {
  json::array rank_hist;
  double mean_q = 0;
  int hasty = 0;
  int agree = 0;
  for (size_t i = 0; i < replies.size(); ++i) {
    const BobReply& r = replies[i];
    if (size_t(r.equity_rank) >= rank_hist.size()) rank_hist.resize(size_t(r.equity_rank) + 1, 0);
    rank_hist[size_t(r.equity_rank)] = rank_hist[size_t(r.equity_rank)].as_int64() + 1;
    mean_q += r.q / double(replies.size());
    hasty += r.equity_rank == 0;
    agree += r.move == rerun[i].move;
  }
  const double n = double(replies.size());
  return {{"hasty_share", hasty / n},
          {"rank_hist", std::move(rank_hist)},
          {"mean_q", mean_q},
          {"rerun_agree", agree / n},
          {"top_replies", top_replies(replies, board)}};
}

}  // namespace

std::string subtree_turn_json(const SubtreeTurn& t) {
  const AliceTurn& turn = *t.turn;
  json::array candidates;
  for (size_t c = 0; c < turn.candidates.size(); ++c) {
    json::object o{{"move", describe_move(turn.candidates[c], turn.position.board)},
                   {"subtree_value", t.subtree_values[c]},
                   {"hasty_value", t.hasty_values[c]},
                   {"searched", bool(t.searched[c])}};
    if (t.searched[c])
      o["replies"] = reply_summary((*t.replies)[c], (*t.rerun)[c], turn.applied[c].board);
    candidates.push_back(std::move(o));
  }
  const json::object out{{"ply", t.ply},
                         {"bag", t.bag},
                         {"bob_racks", turn.bob_racks.size()},
                         {"subtree_pick", argmax(t.subtree_values)},
                         {"hasty_pick", argmax(t.hasty_values)},
                         {"candidates", std::move(candidates)}};
  return json::serialize(out);
}

void append_dump_line(const std::string& path, const std::string& line) {
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);
  std::ofstream(path, std::ios::app) << line << '\n';
}

}  // namespace scribblez
