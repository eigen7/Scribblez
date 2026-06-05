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

void SeedProducer::seed(uint64_t s) { rng_.seed(s); }

uint64_t SeedProducer::next() { return rng_(); }

}  // namespace scribblez
