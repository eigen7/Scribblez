#include "scribblez/seed_producer.h"

namespace scribblez {

SeedProducer::SeedProducer() {
  std::random_device rd;
  rng_.seed((static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd()));
}

SeedProducer& SeedProducer::instance() {
  static SeedProducer p;
  return p;
}

uint64_t SeedProducer::seed(std::optional<uint64_t> requested) {
  uint64_t s;
  if (requested) {
    s = *requested;
  } else {
    std::random_device rd;
    s = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
  }
  rng_.seed(s);
  return s;
}

uint64_t SeedProducer::next() { return rng_(); }

}  // namespace scribblez
