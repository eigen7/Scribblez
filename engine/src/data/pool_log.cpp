#include "data/pool_log.h"

#include "data/sidecar_io.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace scribblez {

namespace {

// Copies `s` into a fixed-width field, NUL-padding the rest; too long throws.
template <size_t N>
void set_field(char (&field)[N], const std::string& s, const char* what) {
  if (s.size() > N) throw util::Exception("PoolWriter: {} '{}' exceeds {} bytes", what, s, N);
  std::memset(field, 0, N);
  std::memcpy(field, s.data(), s.size());
}

// A NUL-padded fixed-width field's string.
template <size_t N>
std::string get_field(const char (&field)[N]) {
  return std::string(field, strnlen(field, N));
}

PoolFileHeader make_header(const PoolSources& s) {
  PoolFileHeader h{};
  h.magic = kPoolMagic;
  h.version = kPoolFormatVersion;
  h.flags = s.leave_model.hash.empty() ? 0 : kPoolFlagLeaveInference;
  h.rollouts = uint32_t(s.rollouts);
  h.leaf_model = s.leaf_model.record("leaf model");
  h.leave_model = s.leave_model.record("leave model");
  return h;
}

PoolOppRack pool_opp_rack(const belief::OppRackSummary& s) {
  PoolOppRack r{};
  r.presence = s.presence;
  r.leave_value_mean = s.leave_value_mean;
  r.leave_value_sd = s.leave_value_sd;
  r.modelled = s.modelled ? 1 : 0;
  return r;
}

}  // namespace

belief::OppRackSummary opp_rack_summary(const PoolOppRack& r) {
  return {.presence = r.presence,
          .modelled = r.modelled != 0,
          .leave_value_mean = r.leave_value_mean,
          .leave_value_sd = r.leave_value_sd};
}

PoolModel PoolModel::of(const PoolModelRecord& r) {
  return {get_field(r.hash), get_field(r.tag), r.generation, r.format_version};
}

PoolModelRecord PoolModel::record(const char* what) const {
  PoolModelRecord r{};
  set_field(r.hash, hash, what);
  set_field(r.tag, tag, what);
  r.generation = generation;
  r.format_version = format_version;
  return r;
}

PoolSources PoolSources::of(const PoolFileHeader& h) {
  return of(h.leaf_model, h.leave_model, int(h.rollouts));
}

PoolSources PoolSources::of(const PoolModelRecord& leaf, const PoolModelRecord& leave,
                            int rollouts) {
  return {
    .leaf_model = PoolModel::of(leaf), .leave_model = PoolModel::of(leave), .rollouts = rollouts};
}

const char* PoolSources::first_difference(const PoolSources& o) const {
  if (leaf_model != o.leaf_model) return "leaf model";
  if (leave_model != o.leave_model) return "leave model";
  if (rollouts != o.rollouts) return "rollout budget";
  return nullptr;
}

PoolWriter::PoolWriter(const std::string& path, const PoolSources& sources) : path_(path) {
  const PoolFileHeader h = make_header(sources);
  append_bytes(&buffer_, &h, sizeof(h));
}

PoolWriter::~PoolWriter() {
  if (!closed_ && std::uncaught_exceptions() == uncaught_at_open_) close();
}

void append_pool_block(std::vector<char>* buffer, const PoolPosition& p, size_t max_values) {
  const size_t k = p.moves.size();
  RELEASE_ASSERT(p.equity_ranks.size() == k && p.labels.size() == k);
  size_t values = 0;
  for (const PoolLabel& l : p.labels) values += size_t(l.rollouts);
  RELEASE_ASSERT(p.rollout_values.size() == values);
  const PoolPositionHeader h{p.at.game_idx, p.at.turn_idx, uint32_t(k), p.num_legal_moves};
  append_bytes(buffer, &h, sizeof(h));
  const PoolOppRack opp_rack = pool_opp_rack(p.opp_rack);
  append_bytes(buffer, &opp_rack, sizeof(opp_rack));
  for (size_t c = 0; c < k; ++c) {
    const PoolLabel& l = p.labels[c];
    const PoolCandidate cand{p.moves[c], float(l.win_prob), float(l.paired_se),
                             float(l.q), p.equity_ranks[c], uint16_t(l.rollouts),
                             0};
    append_bytes(buffer, &cand, sizeof(cand));
  }
  const float* v = p.rollout_values.data();
  for (const PoolLabel& l : p.labels) {
    append_bytes(buffer, v, std::min(size_t(l.rollouts), max_values) * sizeof(float));
    v += l.rollouts;
  }
}

PoolPositionView take_pool_block(const std::vector<char>& buf, size_t* off, size_t max_values,
                                 const std::string& context) {
  PoolPositionView pos{};
  pos.header = reinterpret_cast<const PoolPositionHeader*>(
    take_bytes(buf, off, sizeof(PoolPositionHeader), context));
  pos.opp_rack =
    reinterpret_cast<const PoolOppRack*>(take_bytes(buf, off, sizeof(PoolOppRack), context));
  pos.candidates = reinterpret_cast<const PoolCandidate*>(
    take_bytes(buf, off, pos.header->num_candidates * sizeof(PoolCandidate), context));
  size_t values = 0;
  for (uint32_t c = 0; c < pos.header->num_candidates; ++c)
    values += std::min(size_t(pos.candidates[c].rollouts), max_values);
  pos.rollout_values =
    reinterpret_cast<const float*>(take_bytes(buf, off, values * sizeof(float), context));
  return pos;
}

void PoolWriter::add_position(const PoolPosition& p) {
  RELEASE_ASSERT(!closed_);
  append_pool_block(&buffer_, p, std::numeric_limits<size_t>::max());
  ++num_positions_;
}

void PoolWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  reinterpret_cast<PoolFileHeader*>(buffer_.data())->num_positions = num_positions_;
  write_atomically(path_, buffer_);
}

PoolReader::PoolReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("PoolReader: cannot open {}", path);
  buffer_.resize(size_t(f.tellg()));
  f.seekg(0);
  f.read(buffer_.data(), std::streamsize(buffer_.size()));
  const std::string context = "PoolReader: " + path;
  size_t off = 0;
  take_bytes(buffer_, &off, sizeof(PoolFileHeader), context);
  if (header().magic != kPoolMagic) throw util::Exception("{}: bad magic", context);
  if (header().version != kPoolFormatVersion) {
    throw util::Exception("{}: format version {}, but this build reads {}", context,
                          header().version, kPoolFormatVersion);
  }
  for (uint32_t i = 0; i < header().num_positions; ++i) {
    positions_.push_back(
      take_pool_block(buffer_, &off, std::numeric_limits<size_t>::max(), context));
  }
}

}  // namespace scribblez
