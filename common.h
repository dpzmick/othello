#include <stdint.h>

// mumur3 hash finalizer
static inline uint64_t
hash_u64( uint64_t x )
{
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdUL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53UL;
  x ^= x >> 33;
  return x;
}
