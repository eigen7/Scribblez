#include "data/survivor_log.h"

#include "data/sidecar_io.h"
#include "util/assert.h"
#include "util/exception.h"

#include <fstream>

namespace scribblez {

namespace {

SurvivorRecord record_of(const Survivor& s) {
  return {s.candidate,
          float(s.label.win_prob),
          float(s.label.paired_se),
          float(s.label.q),
          uint16_t(s.label.rollouts),
          uint8_t(s.source),
          0};
}

}  // namespace

SurvivorWriter::SurvivorWriter(const std::string& path, SurvivorReply reply, int rollouts,
                               int solved_racks)
    : path_(path) {
  const SurvivorFileHeader h{kSurvivorMagic,     kSurvivorFormatVersion, uint16_t(reply), 0,
                             uint32_t(rollouts), uint32_t(solved_racks)};
  append_bytes(&buffer_, &h, sizeof(h));
}

SurvivorWriter::~SurvivorWriter() {
  if (!closed_ && std::uncaught_exceptions() == uncaught_at_open_) close();
}

void SurvivorWriter::add_position(const TurnSurvivors& p) {
  RELEASE_ASSERT(!closed_);
  size_t values = 0;
  for (const Survivor& s : p.survivors) values += size_t(s.label.rollouts);
  RELEASE_ASSERT(p.values.size() == values);
  const SurvivorPositionHeader h{p.at.game_idx, p.at.turn_idx, uint32_t(p.survivors.size()), 0};
  append_bytes(&buffer_, &h, sizeof(h));
  for (const Survivor& s : p.survivors) {
    const SurvivorRecord r = record_of(s);
    append_bytes(&buffer_, &r, sizeof(r));
  }
  append_bytes(&buffer_, p.values.data(), values * sizeof(float));
  ++num_positions_;
}

void SurvivorWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  reinterpret_cast<SurvivorFileHeader*>(buffer_.data())->num_positions = num_positions_;
  write_atomically(path_, buffer_);
}

SurvivorReader::SurvivorReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("SurvivorReader: cannot open {}", path);
  buffer_.resize(size_t(f.tellg()));
  f.seekg(0);
  f.read(buffer_.data(), std::streamsize(buffer_.size()));
  const std::string context = "SurvivorReader: " + path;
  size_t off = 0;
  take_bytes(buffer_, &off, sizeof(SurvivorFileHeader), context);
  if (header().magic != kSurvivorMagic) throw util::Exception("{}: bad magic", context);
  if (header().version != kSurvivorFormatVersion) {
    throw util::Exception("{}: format version {}, but this build reads {}", context,
                          header().version, kSurvivorFormatVersion);
  }
  for (uint32_t i = 0; i < header().num_positions; ++i) {
    Position pos{};
    pos.header = reinterpret_cast<const SurvivorPositionHeader*>(
      take_bytes(buffer_, &off, sizeof(SurvivorPositionHeader), context));
    pos.survivors = reinterpret_cast<const SurvivorRecord*>(
      take_bytes(buffer_, &off, pos.header->num_survivors * sizeof(SurvivorRecord), context));
    size_t values = 0;
    for (uint32_t s = 0; s < pos.header->num_survivors; ++s) values += pos.survivors[s].rollouts;
    pos.values =
      reinterpret_cast<const float*>(take_bytes(buffer_, &off, values * sizeof(float), context));
    positions_.push_back(pos);
  }
}

}  // namespace scribblez
