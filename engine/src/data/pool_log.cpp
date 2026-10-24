#include "data/pool_log.h"

#include "data/sidecar_io.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>
#include <fstream>

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

template <size_t H, size_t T>
void set_model(const PoolModel& m, const char* what, char (&hash)[H], char (&tag)[T],
               int32_t* generation, int32_t* format_version) {
  set_field(hash, m.hash, what);
  set_field(tag, m.tag, what);
  *generation = m.generation;
  *format_version = m.format_version;
}

template <size_t H, size_t T>
PoolModel get_model(const char (&hash)[H], const char (&tag)[T], int32_t generation,
                    int32_t format_version) {
  return {get_field(hash), get_field(tag), generation, format_version};
}

PoolFileHeader make_header(const PoolSources& s) {
  PoolFileHeader h{};
  h.magic = kPoolMagic;
  h.version = kPoolFormatVersion;
  h.flags = s.leave_model.hash.empty() ? 0 : kPoolFlagLeaveInference;
  h.rollouts = uint32_t(s.rollouts);
  set_model(s.leaf_model, "leaf model", h.leaf_model_hash, h.leaf_model_tag,
            &h.leaf_model_generation, &h.leaf_model_format_version);
  set_model(s.leave_model, "leave model", h.leave_model_hash, h.leave_model_tag,
            &h.leave_model_generation, &h.leave_model_format_version);
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

PoolSources PoolSources::of(const PoolFileHeader& h) {
  return {.leaf_model = get_model(h.leaf_model_hash, h.leaf_model_tag, h.leaf_model_generation,
                                  h.leaf_model_format_version),
          .leave_model = get_model(h.leave_model_hash, h.leave_model_tag, h.leave_model_generation,
                                   h.leave_model_format_version),
          .rollouts = int(h.rollouts)};
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

void PoolWriter::add_position(const PoolPosition& p) {
  RELEASE_ASSERT(!closed_);
  const size_t k = p.moves.size();
  RELEASE_ASSERT(p.equity_ranks.size() == k && p.labels.size() == k);
  size_t values = 0;
  for (const PoolLabel& l : p.labels) values += size_t(l.rollouts);
  RELEASE_ASSERT(p.rollout_values.size() == values);
  const PoolPositionHeader h{p.at.game_idx, p.at.turn_idx, uint32_t(k), p.num_legal_moves};
  append_bytes(&buffer_, &h, sizeof(h));
  const PoolOppRack opp_rack = pool_opp_rack(p.opp_rack);
  append_bytes(&buffer_, &opp_rack, sizeof(opp_rack));
  for (size_t c = 0; c < k; ++c) {
    const PoolLabel& l = p.labels[c];
    const PoolCandidate cand{p.moves[c], float(l.win_prob), float(l.paired_se),
                             float(l.q), p.equity_ranks[c], uint16_t(l.rollouts),
                             0};
    append_bytes(&buffer_, &cand, sizeof(cand));
  }
  append_bytes(&buffer_, p.rollout_values.data(), p.rollout_values.size() * sizeof(float));
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
    Position pos{};
    pos.header = reinterpret_cast<const PoolPositionHeader*>(
      take_bytes(buffer_, &off, sizeof(PoolPositionHeader), context));
    pos.opp_rack =
      reinterpret_cast<const PoolOppRack*>(take_bytes(buffer_, &off, sizeof(PoolOppRack), context));
    pos.candidates = reinterpret_cast<const PoolCandidate*>(
      take_bytes(buffer_, &off, pos.header->num_candidates * sizeof(PoolCandidate), context));
    size_t values = 0;
    for (uint32_t c = 0; c < pos.header->num_candidates; ++c) values += pos.candidates[c].rollouts;
    pos.rollout_values =
      reinterpret_cast<const float*>(take_bytes(buffer_, &off, values * sizeof(float), context));
    positions_.push_back(pos);
  }
}

}  // namespace scribblez
