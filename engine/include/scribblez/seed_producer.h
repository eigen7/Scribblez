#pragma once

#include <cstdint>
#include <random>

namespace scribblez {

// A process-wide source of 64-bit seeds for any object that owns an internal
// PRNG. By default it is seeded once from std::random_device; call seed() at
// startup (e.g. from a top-level --seed=N option) to make all subsequent
// next() calls deterministic, which in turn makes every object that defaulted
// its RNG to SeedProducer::instance().next() deterministic too.
//
// Per-object overrides still work: an object may take an explicit seed in its
// constructor and bypass the producer entirely.
//
// Caveat: results depend on the *order* of next() calls -- changing the
// construction order of RNG-using objects will change the seed each one
// receives. This is the same caveat as the prior master-mt19937 scheme.
class SeedProducer {
 public:
  static SeedProducer& instance();

  // Re-seed the producer. Call once at startup if --seed was provided.
  void seed(uint64_t s);

  // Produce a fresh 64-bit seed.
  uint64_t next();

 private:
  SeedProducer();
  std::mt19937_64 rng_;
};

}  // namespace scribblez
