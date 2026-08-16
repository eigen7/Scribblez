#include "data/sim_observation_log.h"

#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <unistd.h>

namespace scribblez {

namespace {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  buffer->insert(buffer->end(), p, p + size);
}

}  // namespace

SimObsWriter::SimObsWriter(const std::string& path, uint32_t flags,
                           const std::string& proposer_hash)
    : path_(path) {
  SimObsFileHeader hdr{};
  hdr.magic = kSimObsMagic;
  hdr.version = kSimObsVersion;
  hdr.num_positions = 0;  // patched in close()
  hdr.flags = flags;
  std::memcpy(hdr.proposer_hash, proposer_hash.data(),
              std::min(proposer_hash.size(), sizeof(hdr.proposer_hash)));
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

SimObsWriter::~SimObsWriter() {
  if (!closed_) close();
}

void SimObsWriter::add_position(uint32_t game_index, uint32_t turn_index,
                                const std::vector<Move>& candidates,
                                const std::vector<SimObservation>& observations, uint32_t rollouts,
                                uint64_t base_seed, uint32_t num_legal_moves,
                                uint32_t position_flags) {
  RELEASE_ASSERT(!closed_);
  RELEASE_ASSERT(candidates.size() == observations.size());
  SimObsPositionHeader ph{};
  ph.game_index = game_index;
  ph.turn_index = turn_index;
  ph.num_candidates = uint32_t(candidates.size());
  ph.rollouts = rollouts;
  ph.base_seed = base_seed;
  ph.num_legal_moves = num_legal_moves;
  ph.flags = position_flags;
  append_bytes(&buffer_, &ph, sizeof(ph));
  for (size_t c = 0; c < candidates.size(); ++c) {
    SimObsRecord rec{};
    rec.move = candidates[c];
    rec.obs = observations[c];
    append_bytes(&buffer_, &rec, sizeof(rec));
  }
  ++num_positions_;
}

void SimObsWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  SimObsFileHeader* hdr = reinterpret_cast<SimObsFileHeader*>(buffer_.data());
  hdr->num_positions = num_positions_;
  // Temp-file + rename so the .sobs appears atomically: an interrupted run
  // never leaves a truncated file that a resume (which skips existing
  // sidecars) would silently keep.
  const std::string tmp = std::format("{}.tmp.{}", path_, ::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw util::Exception("SimObsWriter: cannot open {}", tmp);
    f.write(buffer_.data(), std::streamsize(buffer_.size()));
  }
  std::filesystem::rename(tmp, path_);
}

std::string SimObsReader::proposer_hash() const {
  const char* h = header().proposer_hash;
  return std::string(h, strnlen(h, sizeof(header().proposer_hash)));
}

SimObsReader::SimObsReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("SimObsReader: cannot open {}", path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  buffer_.resize(size_t(size));
  f.read(buffer_.data(), size);

  if (buffer_.size() < sizeof(SimObsFileHeader)) {
    throw util::Exception("SimObsReader: truncated header in {}", path);
  }
  const SimObsFileHeader* hdr = reinterpret_cast<const SimObsFileHeader*>(buffer_.data());
  if (hdr->magic != kSimObsMagic) throw util::Exception("SimObsReader: bad magic in {}", path);
  if (hdr->version != kSimObsVersion) {
    throw util::Exception("SimObsReader: version mismatch in {} (file={} code={})", path,
                          hdr->version, kSimObsVersion);
  }

  size_t off = sizeof(SimObsFileHeader);
  positions_.reserve(hdr->num_positions);
  for (uint32_t p = 0; p < hdr->num_positions; ++p) {
    if (off + sizeof(SimObsPositionHeader) > buffer_.size()) {
      throw util::Exception("SimObsReader: truncated position header in {}", path);
    }
    const SimObsPositionHeader* ph =
      reinterpret_cast<const SimObsPositionHeader*>(buffer_.data() + off);
    off += sizeof(SimObsPositionHeader);
    const size_t records_bytes = size_t(ph->num_candidates) * sizeof(SimObsRecord);
    if (off + records_bytes > buffer_.size()) {
      throw util::Exception("SimObsReader: truncated records in {}", path);
    }
    positions_.push_back({ph, reinterpret_cast<const SimObsRecord*>(buffer_.data() + off)});
    off += records_bytes;
  }
}

}  // namespace scribblez
