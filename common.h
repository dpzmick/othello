#pragma once

#include <stdint.h>

#define NS 1000000000UL

#ifdef TARGET_PLAYDATE
#define assert(f)
#define static_assert(e,m) _Static_assert(e,m)
#else
#include <assert.h>
#endif

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

#ifndef TARGET_PLAYDATE

#include <time.h>

static inline uint64_t
wallclock( void )
{
  struct timespec tp[1];
  int ret = clock_gettime( CLOCK_REALTIME, tp );
  if( ret != 0 ) return (uint64_t)-1; // hopefully noticable
  return (uint64_t)tp->tv_sec*NS + (uint64_t)tp->tv_nsec;
}

#endif
