#pragma once

#include <cstdint>
#include <random>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

// A process-wide source of 64-bit seeds for any object owning an internal PRNG.
// It is seeded from std::random_device by default; seeding it at startup makes
// every object that defaulted its RNG to next() deterministic too. An object
// may still take an explicit seed and bypass the producer.
//
// Caveat: results depend on the *order* of next() calls, so changing the
// construction order of RNG-using objects changes the seed each receives.
class SeedProducer {
 public:
  struct Params {
    // 0 means "pick a fresh seed from std::random_device". Yes, that makes an
    // explicit --seed=0 unreachable; that's fine.
    uint64_t seed = 0;

    void add_options(boost::program_options::options_description& desc);
  };

  static SeedProducer& instance();

  // Returns the seed actually used -- `params`', or a freshly generated one.
  uint64_t seed(const Params& params);

  uint64_t next();

 private:
  SeedProducer();
  std::mt19937_64 rng_;
};

}  // namespace scribblez
