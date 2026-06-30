#include "scribblez/scribblez_ffi.h"

#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"
#include "scribblez/data_loader.h"
#include "scribblez/game_runner.h"
#include "scribblez/input_encoder.h"
#include "scribblez/lane_analysis.h"
#include "scribblez/lane_targets.h"
#include "scribblez/max_move_per_lane_input_encoder.h"
#include "scribblez/max_move_per_lane_task.h"
#include "scribblez/player_factory.h"
#include "scribblez/post_move_analysis.h"
#include "scribblez/row_encoder.h"
#include "scribblez/self_play_engine.h"
#include "scribblez/slog_subset.h"
#include "scribblez/streaming_game_producer.h"
#include "scribblez/streaming_row_buffer.h"
#include "scribblez/training_targets.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using scribblez::binlog::DataLoader;
using scribblez::binlog::FileHeader;
using scribblez::binlog::kMagic;
using scribblez::binlog::kVersion;

namespace {

// Static dim arrays referenced by the ScribblezShape entries below. These
// have static storage duration; the addresses returned to the caller stay
// valid for the lifetime of the process.
constexpr int kInputSpatialDims[3] = {scribblez::kSpatialPlanes, scribblez::kBoardSide,
                                      scribblez::kBoardSide};
constexpr int kInputScalarDims[1] = {scribblez::kScalarFloats};

const ScribblezShape kInputShapes[] = {
  {"input_spatial", kInputSpatialDims, 3, -1},
  {"input_scalar", kInputScalarDims, 1, -1},
  {nullptr, nullptr, 0, 0},
};

// Build the (null-terminated) target shape table at compile time directly
// from scribblez::AllTargets, so adding/removing a target struct
// in training_targets.h automatically updates the FFI advertisement with
// no edits here.
template <typename List>
struct TargetShapeTable;

template <typename... Ts>
struct TargetShapeTable<scribblez::TargetList<Ts...>> {
  static constexpr std::size_t kCount = sizeof...(Ts);
  static constexpr std::array<ScribblezShape, kCount + 1> kValue = []() {
    std::array<ScribblezShape, kCount + 1> a{};
    std::size_t i = 0;
    (void)std::initializer_list<int>{
      (a[i] = ScribblezShape{Ts::kName, Ts::kDims, static_cast<int>(std::size(Ts::kDims)),
                             static_cast<int>(i)},
       ++i, 0)...};
    a[kCount] = ScribblezShape{nullptr, nullptr, 0, 0};
    return a;
  }();
};

constexpr auto kTargetShapesArr = TargetShapeTable<scribblez::AllTargets>::kValue;

// --- Max-move-per-lane task shapes (hand-written; the max-move-per-lane encoders are not part of
// the AllTargets pack). Input: 31 board planes + 27 rack scalars. Labels: the
// per-lane occupancy / score / mask blocks of encode_lane_targets. ---
constexpr int kMaxMovePerLaneInputSpatialDims[3] = {
  scribblez::MaxMovePerLaneInputEncoder::kSpatialPlanes, scribblez::kBoardSide,
  scribblez::kBoardSide};
constexpr int kMaxMovePerLaneInputScalarDims[1] = {
  scribblez::MaxMovePerLaneInputEncoder::kScalarFloats};

const ScribblezShape kMaxMovePerLaneInputShapes[] = {
  {"input_spatial", kMaxMovePerLaneInputSpatialDims, 3, -1},
  {"input_scalar", kMaxMovePerLaneInputScalarDims, 1, -1},
  {nullptr, nullptr, 0, 0},
};

constexpr int kMaxMovePerLaneOccupancyDims[3] = {scribblez::kNumLanes, scribblez::kLaneLen,
                                                 scribblez::kLaneTileKinds};
constexpr int kMaxMovePerLaneScoreDims[1] = {scribblez::kNumLanes};
constexpr int kMaxMovePerLaneMaskDims[1] = {scribblez::kNumLanes};

const ScribblezShape kMaxMovePerLaneTargetShapes[] = {
  {"lane_occupancy", kMaxMovePerLaneOccupancyDims, 3, 0},
  {"lane_score", kMaxMovePerLaneScoreDims, 1, 1},
  {"lane_mask", kMaxMovePerLaneMaskDims, 1, 2},
  {nullptr, nullptr, 0, 0},
};

}  // namespace

extern "C" {

const ScribblezShape* scribblez_input_shapes(void) { return kInputShapes; }
const ScribblezShape* scribblez_target_shapes(void) { return kTargetShapesArr.data(); }

int scribblez_row_size_floats(void) { return DataLoader::row_size_floats(); }

int scribblez_input_floats(void) { return scribblez::kInputFloats; }

const ScribblezShape* scribblez_max_move_per_lane_input_shapes(void) {
  return kMaxMovePerLaneInputShapes;
}
const ScribblezShape* scribblez_max_move_per_lane_target_shapes(void) {
  return kMaxMovePerLaneTargetShapes;
}

int scribblez_max_move_per_lane_row_size_floats(void) {
  return scribblez::MaxMovePerLaneTask::kRowFloats;
}

int scribblez_max_move_per_lane_input_floats(void) {
  return scribblez::MaxMovePerLaneInputEncoder::kInputFloats;
}

namespace {

// Read an entire .slog file into `buf`, validate its header, and report the
// game count. When `game_idx >= 0`, also bounds-check it. Returns 0 on
// success, -1 on any failure.
int load_slog(const char* path, int64_t game_idx, std::vector<char>& buf, uint32_t* num_games) {
  if (!path) return -1;
  std::ifstream f(path, std::ios::binary);
  if (!f) return -1;
  buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < sizeof(FileHeader)) return -1;
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  if (hdr->magic != kMagic || hdr->version != kVersion) return -1;
  if (game_idx >= static_cast<int64_t>(hdr->num_games)) return -1;
  if (num_games) *num_games = hdr->num_games;
  return 0;
}

}  // namespace

int scribblez_encode_score_diff_sweep(const char* path, int64_t game_idx, int post_move,
                                      int diff_lo, int diff_hi, float* out_inputs) {
  if (!out_inputs || diff_hi < diff_lo) return -1;
  uint32_t num_games = 0;
  std::vector<char> buf;
  if (load_slog(path, game_idx, buf, &num_games) != 0) return -1;

  const int64_t sweep = static_cast<int64_t>(diff_hi - diff_lo + 1) * scribblez::kInputFloats;
  scribblez::binlog::BlockDecoder decoder;
  if (game_idx >= 0) {
    // A single position.
    decoder.encode_score_diff_sweep(buf.data(), static_cast<uint32_t>(game_idx), post_move != 0,
                                    diff_lo, diff_hi, out_inputs);
  } else {
    // Every position in the file, position-major (game g at row g * R).
    for (uint32_t g = 0; g < num_games; ++g) {
      decoder.encode_score_diff_sweep(buf.data(), g, post_move != 0, diff_lo, diff_hi,
                                      out_inputs + static_cast<int64_t>(g) * sweep);
    }
  }
  return 0;
}

namespace {

// Copy `s` into the caller's NUL-terminated buffer (truncating to out_cap) and
// return the full length, which may exceed out_cap - 1 (caller should retry).
int emit_string(const std::string& s, char* out, int out_cap) {
  const int len = static_cast<int>(s.size());
  if (out && out_cap > 0) {
    const int n = len < out_cap - 1 ? len : out_cap - 1;
    std::memcpy(out, s.data(), static_cast<size_t>(n));
    out[n] = '\0';
  }
  return len;
}

}  // namespace

int scribblez_dump_position(const char* path, int64_t game_idx, int post_move, char* out,
                            int out_cap) {
  std::vector<char> buf;
  if (load_slog(path, game_idx, buf, nullptr) != 0) return -1;
  scribblez::binlog::BlockDecoder decoder;
  return emit_string(
    decoder.dump_position(buf.data(), static_cast<uint32_t>(game_idx), post_move != 0), out,
    out_cap);
}

int scribblez_dump_position_json(const char* path, int64_t game_idx, int post_move, char* out,
                                 int out_cap) {
  std::vector<char> buf;
  if (load_slog(path, game_idx, buf, nullptr) != 0) return -1;
  scribblez::binlog::BlockDecoder decoder;
  return emit_string(
    decoder.dump_position_json(buf.data(), static_cast<uint32_t>(game_idx), post_move != 0), out,
    out_cap);
}

int scribblez_max_move_per_lane_analyze_gcg(const char* gcg_text, char* out_json, int out_cap,
                                            float* out_input) {
  if (!gcg_text) return -1;
  try {
    scribblez::GcgAnalysisPosition pos;
    std::string error;
    if (!scribblez::parse_gcg_analysis_position(gcg_text, &pos, &error)) return -1;
    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    if (out_input)
      scribblez::MaxMovePerLaneInputEncoder::encode(pos.board, pos.rack, /*flip=*/false, out_input);
    return emit_string(scribblez::lane_analysis_json(pos.board, pos.rack, pos.on_move, dict),
                       out_json, out_cap);
  } catch (const std::exception&) {
    return -1;
  }
}

int scribblez_post_move_value_analyze_gcg(const char* gcg_text, float* out_input) {
  if (!gcg_text || !out_input) return -1;
  try {
    std::string error;
    if (!scribblez::encode_post_move_analysis_input(gcg_text, out_input, &error)) return -1;
    return scribblez::kInputFloats;
  } catch (const std::exception&) {
    return -1;
  }
}

int scribblez_post_move_value_board_json(const char* gcg_text, char* out_json, int out_cap) {
  if (!gcg_text) return -1;
  try {
    std::string error;
    const std::string json = scribblez::post_move_analysis_board_json(gcg_text, &error);
    if (json.empty()) return -1;
    return emit_string(json, out_json, out_cap);
  } catch (const std::exception&) {
    return -1;
  }
}

int scribblez_sample_slog(const char* dst_path, const char* const* src_paths,
                          const int64_t* game_indices, int num_picks) {
  if (!dst_path || !src_paths || !game_indices || num_picks < 0) return -1;
  std::vector<scribblez::binlog::SlogPick> picks;
  picks.reserve(static_cast<size_t>(num_picks));
  for (int i = 0; i < num_picks; ++i) {
    if (!src_paths[i]) return -1;
    picks.push_back({src_paths[i], game_indices[i]});
  }
  return scribblez::binlog::write_slog_subset(dst_path, picks) ? 0 : -1;
}

int scribblez_read_file_header(const char* path, int64_t* out_num_positions,
                               int64_t* out_file_size) {
  if (!path || !out_num_positions || !out_file_size) return -1;
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  FileHeader hdr{};
  const size_t n = std::fread(&hdr, sizeof(hdr), 1, f);
  std::fclose(f);
  if (n != 1) return -1;
  if (hdr.magic != kMagic) return -1;
  if (hdr.version != kVersion) return -1;
  std::error_code ec;
  const auto fsz = std::filesystem::file_size(path, ec);
  if (ec) return -1;
  *out_num_positions = static_cast<int64_t>(hdr.num_games);
  *out_file_size = static_cast<int64_t>(fsz);
  return 0;
}

struct DataLoaderHandle {
  DataLoader loader;
  explicit DataLoaderHandle(const DataLoader::Params& p) : loader(p) {}
};

DataLoaderHandle* scribblez_dl_new(int64_t memory_budget, int num_worker_threads,
                                   int num_prefetch_threads) {
  DataLoader::Params p;
  p.memory_budget = memory_budget;
  p.num_worker_threads = num_worker_threads;
  p.num_prefetch_threads = num_prefetch_threads;
  return new DataLoaderHandle(p);
}

void scribblez_dl_delete(DataLoaderHandle* h) { delete h; }

void scribblez_dl_add_file(DataLoaderHandle* h, const char* path, int64_t num_positions,
                           int64_t file_size) {
  if (!h || !path) return;
  h->loader.add_file(path, num_positions, file_size);
}

int64_t scribblez_dl_num_positions(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.num_positions();
}

int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed, int turns_per_game, int epoch_index) {
  if (!h) return 0;
  DataLoader::EpochConfig cfg;
  cfg.batch_size = batch_size;
  cfg.post_move = post_move != 0;
  cfg.apply_symmetry = apply_symmetry != 0;
  cfg.seed = seed;
  cfg.turns_per_game = turns_per_game;
  cfg.epoch_index = epoch_index;
  return h->loader.epoch_start(cfg);
}

int scribblez_dl_load_batch(DataLoaderHandle* h, float* output) {
  if (!h || !output) return 0;
  return h->loader.load_batch(output);
}

int64_t scribblez_dl_resident_bytes(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.resident_bytes();
}

// ---------------------------------------------------------------------------
// Streaming self-play -> training pipeline
// ---------------------------------------------------------------------------

struct StreamHandle {
  scribblez::binlog::StreamingRowBuffer ring;
  scribblez::binlog::StreamingGameProducer producer;

  StreamHandle(float* const* slots, int num_slots, int rows_per_slot, int row_floats,
               const scribblez::SelfPlayEngine::Params& engine_params,
               const scribblez::PlayerFactory::Params& player_params,
               const scribblez::binlog::StreamingGameProducer::Params& stream_params)
      : ring(slots, num_slots, rows_per_slot, row_floats),
        producer(engine_params, player_params, stream_params, ring) {}
};

namespace {

// Shared construction for both streaming entry points: validate the config,
// build the engine + player params, and create the StreamHandle with the given
// row width and per-worker encoder factory. Returns nullptr on bad config or a
// lexicon-load failure. Both tasks play the same HastyBot self-play games; only
// the per-row encoding (and thus the row width) differs.
StreamHandle* new_stream(float* const* slot_ptrs, int num_slots, int rows_per_slot, int num_threads,
                         int apply_symmetry, uint64_t seed, int handicap_max,
                         const char* const* player_specs, int num_specs, int row_floats,
                         scribblez::binlog::RowEncoderFactory factory) {
  if (!slot_ptrs || num_slots < 1 || rows_per_slot < 1 || !player_specs || num_specs < 1) {
    return nullptr;
  }
  scribblez::PlayerFactory::Params player_params;
  for (int i = 0; i < num_specs; ++i) {
    if (!player_specs[i]) return nullptr;
    player_params.specs.emplace_back(player_specs[i]);
  }
  scribblez::SelfPlayEngine::Params engine_params;
  engine_params.threads = num_threads;
  engine_params.seed = seed;
  engine_params.handicap_max = handicap_max;
  scribblez::binlog::StreamingGameProducer::Params stream_params;
  stream_params.apply_symmetry = apply_symmetry != 0;
  stream_params.make_encoder = std::move(factory);
  try {
    // Surface a missing-lexicon error here (at construction) rather than mid-game.
    scribblez::GameRunner::load_dictionary_or_throw();
    return new StreamHandle(slot_ptrs, num_slots, rows_per_slot, row_floats, engine_params,
                            player_params, stream_params);
  } catch (const std::exception& e) {
    std::cerr << "new_stream: " << e.what() << "\n";
    return nullptr;
  }
}

}  // namespace

StreamHandle* scribblez_stream_new(float* const* slot_ptrs, int num_slots, int rows_per_slot,
                                   int num_threads, int post_move, int apply_symmetry,
                                   uint64_t seed, int handicap_max, const char* const* player_specs,
                                   int num_specs) {
  const bool pm = post_move != 0;
  return new_stream(slot_ptrs, num_slots, rows_per_slot, num_threads, apply_symmetry, seed,
                    handicap_max, player_specs, num_specs,
                    scribblez::binlog::DataLoader::row_size_floats(),
                    [pm]() { return scribblez::binlog::make_post_move_value_row_encoder(pm); });
}

StreamHandle* scribblez_max_move_per_lane_stream_new(
  float* const* slot_ptrs, int num_slots, int rows_per_slot, int num_threads, int apply_symmetry,
  uint64_t seed, int handicap_max, const char* const* player_specs, int num_specs) {
  // The max-move-per-lane encoder enumerates legal moves, so it needs the lexicon. Bind it
  // once (a process-static reference) into the per-worker encoder factory.
  const scribblez::Dictionary* dict = nullptr;
  try {
    dict = &scribblez::GameRunner::load_dictionary_or_throw();
  } catch (const std::exception& e) {
    std::cerr << "scribblez_max_move_per_lane_stream_new: " << e.what() << "\n";
    return nullptr;
  }
  return new_stream(
    slot_ptrs, num_slots, rows_per_slot, num_threads, apply_symmetry, seed, handicap_max,
    player_specs, num_specs, scribblez::MaxMovePerLaneTask::kRowFloats,
    [dict]() { return scribblez::binlog::make_max_move_per_lane_row_encoder(*dict); });
}

void scribblez_stream_start(StreamHandle* h) {
  if (h) h->producer.start();
}

int scribblez_stream_wait_full_slot(StreamHandle* h) {
  if (!h) return -1;
  return h->ring.wait_full_slot();
}

void scribblez_stream_release_slot(StreamHandle* h, int slot) {
  if (h) h->ring.release_slot(slot);
}

void scribblez_stream_get_stats(StreamHandle* h, ScribblezStreamStats* out) {
  if (!h || !out) return;
  const scribblez::binlog::ProducerStats ps = h->producer.stats();
  const scribblez::binlog::RingStats rs = h->ring.stats();
  out->games_played = ps.games_played;
  out->games_dropped = ps.games_dropped;
  out->rows_committed = rs.rows_committed;
  out->slots_published = rs.slots_published;
  out->producer_blocked_ns = rs.producer_blocked_ns;
  out->consumer_blocked_ns = rs.consumer_blocked_ns;
}

void scribblez_stream_stop(StreamHandle* h) {
  if (h) h->producer.stop();
}

void scribblez_stream_delete(StreamHandle* h) { delete h; }

}  // extern "C"
