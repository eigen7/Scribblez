#include "selfplay/sim_observation_log.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace scribblez {

namespace {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  buffer->insert(buffer->end(), p, p + size);
}

}  // namespace

SimObsWriter::SimObsWriter(const std::string& path, uint32_t flags) : path_(path) {
  SimObsFileHeader hdr{};
  hdr.magic = kSimObsMagic;
  hdr.version = kSimObsVersion;
  hdr.num_positions = 0;  // patched in close()
  hdr.flags = flags;
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

SimObsWriter::~SimObsWriter() {
  if (!closed_) close();
}

void SimObsWriter::add_position(uint32_t game_index, uint32_t turn_index,
                                const std::vector<Move>& candidates,
                                const std::vector<SimObservation>& observations, uint32_t rollouts,
                                uint64_t base_seed) {
  assert(!closed_);
  assert(candidates.size() == observations.size());
  SimObsPositionHeader ph{};
  ph.game_index = game_index;
  ph.turn_index = turn_index;
  ph.num_candidates = static_cast<uint32_t>(candidates.size());
  ph.rollouts = rollouts;
  ph.base_seed = base_seed;
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
  assert(!closed_);
  closed_ = true;
  SimObsFileHeader* hdr = reinterpret_cast<SimObsFileHeader*>(buffer_.data());
  hdr->num_positions = num_positions_;
  // Temp-file + rename so the .sobs appears atomically: an interrupted run
  // never leaves a truncated file that a resume (which skips existing
  // sidecars) would silently keep.
  const std::string tmp = path_ + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("SimObsWriter: cannot open " + tmp);
    f.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
  }
  std::filesystem::rename(tmp, path_);
}

SimObsReader::SimObsReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("SimObsReader: cannot open " + path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  buffer_.resize(static_cast<size_t>(size));
  f.read(buffer_.data(), size);

  if (buffer_.size() < sizeof(SimObsFileHeader)) {
    throw std::runtime_error("SimObsReader: truncated header in " + path);
  }
  const SimObsFileHeader* hdr = reinterpret_cast<const SimObsFileHeader*>(buffer_.data());
  if (hdr->magic != kSimObsMagic) throw std::runtime_error("SimObsReader: bad magic in " + path);
  if (hdr->version != kSimObsVersion) {
    throw std::runtime_error("SimObsReader: version mismatch in " + path +
                             " (file=" + std::to_string(hdr->version) +
                             " code=" + std::to_string(kSimObsVersion) + ")");
  }

  size_t off = sizeof(SimObsFileHeader);
  positions_.reserve(hdr->num_positions);
  for (uint32_t p = 0; p < hdr->num_positions; ++p) {
    if (off + sizeof(SimObsPositionHeader) > buffer_.size()) {
      throw std::runtime_error("SimObsReader: truncated position header in " + path);
    }
    const SimObsPositionHeader* ph =
      reinterpret_cast<const SimObsPositionHeader*>(buffer_.data() + off);
    off += sizeof(SimObsPositionHeader);
    const size_t records_bytes = static_cast<size_t>(ph->num_candidates) * sizeof(SimObsRecord);
    if (off + records_bytes > buffer_.size()) {
      throw std::runtime_error("SimObsReader: truncated records in " + path);
    }
    positions_.push_back({ph, reinterpret_cast<const SimObsRecord*>(buffer_.data() + off)});
    off += records_bytes;
  }
}

}  // namespace scribblez
