#include "sim/candidate_race.h"

#include "sim/rollout_report.h"

#include <algorithm>

namespace scribblez {

std::vector<int> race_checkpoints(int rollouts) {
  std::vector<int> out;
  for (const int pct : {10, 20, 40, 70})
    if (rollouts * pct / 100 > 0) out.push_back(rollouts * pct / 100);
  return out;
}

size_t race_leader(const std::vector<size_t>& alive, const CandidateRollouts& rollouts) {
  size_t leader = alive.front();
  double best = -1;
  for (const size_t c : alive) {
    double wins = 0;
    for (const Rollout& r : rollouts[c]) wins += r.win_value();
    if (wins > best) {
      best = wins;
      leader = c;
    }
  }
  return leader;
}

void stop_the_beaten(const CandidateRollouts& rollouts, double sigmas, int protected_count,
                     std::vector<size_t>* alive) {
  const std::vector<Rollout>& lead = rollouts[race_leader(*alive, rollouts)];
  std::erase_if(*alive, [&](size_t c) {
    if (c < size_t(protected_count)) return false;
    return clearly_below(paired_win_diff(rollouts[c], lead), lead.size(), sigmas);
  });
}

}  // namespace scribblez
