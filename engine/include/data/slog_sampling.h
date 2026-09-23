#pragma once

// Position sampling and input handling shared by the tools that generate
// sidecar files for a .slog (sim_obs_tool, move_set_eval_target_generator,
// evidence_trajectory_generator, and others).
//
// Sampling depends only on the run seed and the game, never on thread count or
// visit order, and every tool uses the same per-game shuffle. So for equal
// seeds, a tool sampling k positions per game picks a subset of what a tool
// sampling k' > k picks. The move-set target generator relies on this to find
// an evidence-trajectory sidecar's positions among its own (docs/roadmap.md
// item 4) with no coordination beyond the seed.

#include "data/binary_log.h"

#include <compare>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

// A (game, turn) within a .slog file, ordered as the file stores them.
struct GamePositionIndex {
  uint32_t game_idx;
  uint32_t turn_idx;

  auto operator<=>(const GamePositionIndex&) const = default;
};

// The seed for a position's own randomness (rollout seeds, trajectory draws,
// and the like), identical across all sidecar generators.
inline uint64_t position_seed(uint64_t run_seed, uint32_t game_idx, uint32_t turn_idx);

// Appends `positions_per_game` of the game's eligible turns to `out`, sampled
// without replacement. `positions_per_game` <= 0 takes every eligible turn,
// in order.
void sample_eligible_turns(const GameMetadata& gm, uint32_t game_idx, uint64_t run_seed,
                           int positions_per_game, std::vector<GamePositionIndex>* out);

// The number of turns sample_eligible_turns would append.
int count_eligible_sample(const GameMetadata& gm, int positions_per_game);

// count_eligible_sample summed over the first `limit_games` games of a loaded
// .slog (all games if limit_games <= 0).
uint64_t count_sampled_positions(const std::vector<char>& buf, int positions_per_game,
                                 int limit_games);

// An input .slog that has no sidecar yet, with its contents. The tools load
// all pending files up front so the total work is known before it starts.
struct PendingSlog {
  std::filesystem::path path;
  std::vector<char> bytes;

  std::filesystem::path sidecar(const char* ext) const {
    return std::filesystem::path(path).replace_extension(ext);
  }
};

// Throws util::CleanException naming the file if it cannot be opened.
std::vector<char> read_file_bytes(const std::filesystem::path& path);

// The sidecar generators' shared input convention: the --slog-file arguments
// if any, else every .slog in --slog-dir, sorted. Throws util::CleanException
// if neither is given or no file matches.
std::vector<std::filesystem::path> resolve_slog_inputs(const std::string& slog_dir,
                                                       const std::vector<std::string>& slog_files);

// Loads every input that has no `sidecar_ext` sidecar yet, skipping (with a
// note on stderr) files with a bad header. A face-up-leaves .slog when
// !accept_face_up throws util::CleanException, with `face_up_error` formatted
// with the file name so each tool can name its own remedy.
std::vector<PendingSlog> load_pending_slogs(const std::vector<std::filesystem::path>& slogs,
                                            const char* sidecar_ext, bool accept_face_up,
                                            const char* face_up_error);

}  // namespace binlog
}  // namespace scribblez

#include "inlines/data/slog_sampling.inl"
