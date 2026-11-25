#include "data/belief_log.h"

#include "belief/leave_draws.h"
#include "belief/leave_query.h"
#include "belief/slog_leave_beliefs.h"
#include "data/forgone_play_log.h"
#include "data/sidecar_io.h"
#include "lexicon/leave_value_arrays.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/work_pool.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdfloat>

namespace scribblez {

namespace {

using belief::LeaveQuery;

uint16_t to_half(double x) { return std::bit_cast<uint16_t>(std::float16_t(float(x))); }
double from_half(uint16_t bits) { return double(std::bit_cast<std::float16_t>(bits)); }

template <size_t N>
void copy_text(char (&dst)[N], const std::string& text) {
  std::memset(dst, 0, N);
  std::memcpy(dst, text.data(), std::min(text.size(), N));
}

template <size_t N>
std::string read_text(const char (&field)[N]) {
  return std::string(field, strnlen(field, N));
}

template <size_t N>
bool text_equals(const char (&field)[N], const std::string& text) {
  char expected[N];
  copy_text(expected, text);
  return std::memcmp(field, expected, N) == 0;
}

const binlog::GameMetadata* slog_metadata(const char* slog) {
  return reinterpret_cast<const binlog::GameMetadata*>(slog + sizeof(binlog::FileHeader));
}

uint32_t slog_num_games(const char* slog) {
  return reinterpret_cast<const binlog::FileHeader*>(slog)->num_games;
}

// The entries' turns of a game: the leave inference rows' region.
int region_begin(const binlog::GameMetadata& m) { return m.eligible_begin; }
int region_size(const binlog::GameMetadata& m) {
  return std::max(0, int(m.eligible_end) - int(m.eligible_begin));
}

// A leave with no model's belief: empty (nothing yet, or nothing kept), or
// the seven tiles kept by a pass, which the leave table does not hold. A
// pass's expected counts are seven times each kind's share of the pool; its
// value moments are zero, the cause saying why.
LeaveBelief unmodelled_belief(const LeaveQuery& q) {
  RELEASE_ASSERT(q.length == 0 || q.length > kLeaveArraysMaxLength,
                 "an unmodelled leave of a length the table holds");
  LeaveBelief b;
  b.cause = q.cause;
  b.length = q.length;
  const int pool = q.leave_pool.size();
  if (q.length > 0 && pool > 0) {
    for (int t = 0; t < TILE_KINDS; ++t)
      b.expected[size_t(t)] = double(q.length) * q.leave_pool.count(Tile::of(t)) / pool;
  }
  return b;
}

LeaveBelief modelled_belief(const LeaveQuery& q, const belief::LeaveParams& params,
                            const belief::LeaveSampler& sampler) {
  const belief::LeaveMoments m = sampler.moments();
  LeaveBelief b;
  b.cause = q.cause;
  b.length = q.length;
  b.params = params;
  b.value_mean = m.value_mean;
  b.value_sd = m.value_sd;
  std::copy(m.expected.begin(), m.expected.end(), b.expected.begin());
  return b;
}

// One turn's entry, and its stored draws when `draws` > 0.
struct Annotated {
  BeliefEntry entry;
  std::vector<Rack> draws;
};

Annotated annotate(const LeaveQuery& q, const belief::LeaveParams& params,
                   const LeaveValueArrays& table, int draws, uint64_t seed) {
  std::shared_ptr<const belief::LeaveSampler> sampler;
  if (q.modelled()) sampler = belief::belief_of(q, params, table);
  Annotated out;
  out.entry = (sampler ? modelled_belief(q, params, *sampler) : unmodelled_belief(q)).pack();
  // The draws read the model's belief, or the hypergeometric prior where
  // DeepBot would draw from one (belief::draws_read_belief).
  if (draws > 0) {
    if (!sampler && belief::draws_read_belief(q))
      sampler = belief::belief_of(q, belief::LeaveParams{}, table);
    out.draws = belief::opp_leave_draws(q, sampler.get(), draws, seed);
  }
  return out;
}

// Every game's turns' queries, in file order.
std::vector<LeaveQuery> all_queries(const belief::SlogLeaveBeliefs& beliefs, const char* slog,
                                    const std::vector<uint32_t>& first) {
  const binlog::GameMetadata* metas = slog_metadata(slog);
  std::vector<LeaveQuery> out(first.back());
  util::parallel_lanes(first.size() - 1, [&](util::LaneCursor& cursor) {
    for (size_t g; cursor.next(g);) {
      for (int k = 0; k < region_size(metas[g]); ++k)
        out[first[g] + size_t(k)] =
          beliefs.query({uint32_t(g), uint32_t(region_begin(metas[g]) + k)});
    }
  });
  return out;
}

BeliefFileHeader file_header(const char* slog, const LeaveValueArrays& table,
                             const nn::ModelIdentity& id, int draws) {
  BeliefFileHeader h{};
  h.magic = kBeliefMagic;
  h.version = kBeliefVersion;
  h.draws = uint16_t(draws);
  h.num_games = slog_num_games(slog);
  copy_text(h.model_hash, id.hash);
  copy_text(h.model_tag, id.tag);
  h.model_generation = id.generation;
  h.model_format_version = id.format_version;
  copy_text(h.leave_values_hash, table.content_hash());
  copy_text(h.slog_binding, forgone_slog_binding(slog));
  return h;
}

void check_header(const std::string& path, const BeliefFileHeader& h, const char* slog,
                  const std::string& leave_values_hash) {
  if (h.magic != kBeliefMagic) throw util::Exception("{}: not a .belief file", path);
  if (h.version != kBeliefVersion) {
    throw util::Exception("{}: version {}, expected {}", path, h.version, kBeliefVersion);
  }
  if (h.num_games != slog_num_games(slog)) {
    throw util::Exception("{}: {} games, its .slog has {}", path, h.num_games,
                          slog_num_games(slog));
  }
  if (!text_equals(h.leave_values_hash, leave_values_hash)) {
    throw util::Exception("{}: summarized over other leave values", path);
  }
  if (!text_equals(h.slog_binding, forgone_slog_binding(slog))) {
    throw util::Exception("{}: does not match its .slog (regenerated since?)", path);
  }
}

}  // namespace

BeliefEntry LeaveBelief::pack() const {
  BeliefEntry e{};
  e.cause = uint8_t(cause);
  e.length = uint8_t(length);
  for (int t = 0; t < TILE_KINDS; ++t) e.params[size_t(t)] = to_half(params.theta[size_t(t)]);
  e.params[TILE_KINDS] = to_half(params.a);
  e.params[TILE_KINDS + 1] = to_half(params.b);
  e.value_mean = to_half(value_mean);
  e.value_sd = to_half(value_sd);
  for (int t = 0; t < TILE_KINDS; ++t) {
    const double share = length > 0 ? expected[size_t(t)] / length : 0.0;
    e.expected[size_t(t)] = uint8_t(std::lround(std::clamp(share, 0.0, 1.0) * 255.0));
  }
  return e;
}

LeaveBelief LeaveBelief::unpack(const BeliefEntry& e) {
  LeaveBelief b;
  b.cause = LeaveRowCause(e.cause);
  b.length = e.length;
  for (int t = 0; t < TILE_KINDS; ++t)
    b.params.theta[size_t(t)] = float(from_half(e.params[size_t(t)]));
  b.params.a = float(from_half(e.params[TILE_KINDS]));
  b.params.b = float(from_half(e.params[TILE_KINDS + 1]));
  b.value_mean = from_half(e.value_mean);
  b.value_sd = from_half(e.value_sd);
  for (int t = 0; t < TILE_KINDS; ++t)
    b.expected[size_t(t)] = double(e.expected[size_t(t)]) / 255.0 * b.length;
  return b;
}

uint64_t belief_draw_seed(uint32_t game, int turn) {
  return util::splitmix64((uint64_t(game) << 16) | uint64_t(turn));
}

std::string belief_path_of(const std::string& slog_path) {
  return std::filesystem::path(slog_path).replace_extension(kBeliefExtension).string();
}

void write_belief_sidecar(const std::vector<char>& slog, const std::string& slog_path,
                          const Dictionary& dict, const LeaveValueArrays& table,
                          const BeliefModel& model, int draws, const std::string& path) {
  const uint32_t num_games = slog_num_games(slog.data());
  const binlog::GameMetadata* metas = slog_metadata(slog.data());
  std::vector<uint32_t> first(size_t(num_games) + 1, 0);
  for (uint32_t g = 0; g < num_games; ++g)
    first[g + 1] = first[g] + uint32_t(region_size(metas[g]));

  const belief::SlogLeaveBeliefs beliefs(slog, slog_path, dict, table, model.predict);
  const std::vector<LeaveQuery> queries = all_queries(beliefs, slog.data(), first);
  const std::vector<belief::LeaveParams> params = belief::predict_beliefs(queries, model.predict);

  std::vector<Annotated> annotated(queries.size());
  util::parallel_lanes(num_games, [&](util::LaneCursor& cursor) {
    for (size_t g; cursor.next(g);) {
      for (uint32_t i = first[g]; i < first[g + 1]; ++i) {
        const int turn = region_begin(metas[g]) + int(i - first[g]);
        annotated[i] =
          annotate(queries[i], params[i], table, draws, belief_draw_seed(uint32_t(g), turn));
      }
    }
  });

  const BeliefFileHeader h = file_header(slog.data(), table, model.identity, draws);
  std::vector<char> bytes;
  append_bytes(&bytes, &h, sizeof(h));
  append_bytes(&bytes, first.data(), first.size() * sizeof(uint32_t));
  for (const Annotated& a : annotated) append_bytes(&bytes, &a.entry, sizeof(a.entry));
  for (const Annotated& a : annotated) {
    for (const Rack& r : a.draws) append_bytes(&bytes, r.tiles().data(), RACK_SIZE);
  }
  write_atomically(path, bytes);
}

BeliefSidecar::BeliefSidecar(const std::string& path, const char* slog,
                             const std::string& leave_values_hash) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw util::Exception("{}: missing (every .slog needs its .belief sidecar)", path);
  bytes_.assign(std::istreambuf_iterator<char>(in), {});
  if (bytes_.size() < sizeof(BeliefFileHeader)) throw util::Exception("{}: truncated header", path);
  check_header(path, header(), slog, leave_values_hash);
  if (bytes_.size() < entries_offset()) throw util::Exception("{}: truncated index", path);
  const size_t entries = first_entry()[header().num_games];
  const size_t expected =
    entries_offset() + entries * sizeof(BeliefEntry) + entries * size_t(header().draws) * RACK_SIZE;
  if (bytes_.size() != expected) throw util::Exception("{}: body does not match its index", path);
}

nn::ModelIdentity BeliefSidecar::identity() const {
  const BeliefFileHeader& h = header();
  return {read_text(h.model_hash), read_text(h.model_tag), h.model_generation,
          h.model_format_version};
}

std::span<const BeliefEntry> BeliefSidecar::game(uint32_t g) const {
  RELEASE_ASSERT(g < header().num_games);
  const auto* entries = reinterpret_cast<const BeliefEntry*>(bytes_.data() + entries_offset());
  return {entries + first_entry()[g], entries + first_entry()[g + 1]};
}

Rack BeliefSidecar::draw(uint32_t g, size_t k, int i) const {
  RELEASE_ASSERT(draws() > 0 && i >= 0 && i < draws());
  RELEASE_ASSERT(k < game(g).size());
  const size_t entries = first_entry()[header().num_games];
  const size_t entry = first_entry()[g] + k;
  const char* tiles = bytes_.data() + entries_offset() + entries * sizeof(BeliefEntry) +
                      (entry * size_t(draws()) + size_t(i)) * RACK_SIZE;
  Rack r;
  for (int j = 0; j < RACK_SIZE; ++j) {
    const Tile t = std::bit_cast<Tile>(tiles[j]);
    if (!t.is_empty()) r.add(t);
  }
  return r;
}

const BeliefFileHeader& BeliefSidecar::header() const {
  return *reinterpret_cast<const BeliefFileHeader*>(bytes_.data());
}

const uint32_t* BeliefSidecar::first_entry() const {
  return reinterpret_cast<const uint32_t*>(bytes_.data() + sizeof(BeliefFileHeader));
}

size_t BeliefSidecar::entries_offset() const {
  return sizeof(BeliefFileHeader) + (size_t(header().num_games) + 1) * sizeof(uint32_t);
}

}  // namespace scribblez
