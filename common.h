#include <stdint.h>

#define NS 1000000000UL

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

// FIXME not going to be supported on the actual target
#include <time.h>

static inline uint64_t
wallclock( void )
{
  struct timespec tp[1];
  int ret = clock_gettime( CLOCK_REALTIME, tp );
  if( ret != 0 ) return (uint64_t)-1; // hopefully noticable
  return (uint64_t)tp->tv_sec*NS + (uint64_t)tp->tv_nsec;
}
