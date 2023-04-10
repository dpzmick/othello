#pragma once

#include <stdint.h>
#include <time.h>

#define NS 1000000000UL

#define ARRAY_SIZE(a) (sizeof((a))/sizeof(*(a)))
#define LIKELY(c)   __builtin_expect(c, 1)
#define UNLIKELY(c) __builtin_expect(c, 0)
#define MAX(a, b) ((a) < (b) ? (b) : (a))

void
__attribute__((noreturn))
__attribute__((cold))
__attribute__((format (printf, 3, 4 )))
_fail( char const * file,
       int          line,
       char const * fmt, ... );

#define Fail( ... ) _fail( __FILE__, __LINE__, __VA_ARGS__ )

static inline uint64_t
wallclock( void )
{
  struct timespec tp[1];
  int ret = clock_gettime( CLOCK_REALTIME, tp );
  if( ret != 0 ) return (uint64_t)-1; // hopefully noticable
  return (uint64_t)tp->tv_sec*NS + (uint64_t)tp->tv_nsec;
}

static inline uint64_t
next_pow2( uint64_t x )
{
  // count leading zeros.. 0000001.... returns 6
  if( x<2 ) return 2; // manage overflow and UB w/ clzll(0)
  uint64_t first_one = 64ul - (uint64_t)__builtin_clzll( x-1 ); // -1 in case number is already a power of two
  return 1ul << first_one;
}

// UB if popcount(bitset) < idx
static inline uint64_t
keep_ith_set_bit( uint64_t bitset,
                  uint64_t idx )
{
  while( idx ) {
    uint64_t offset = UINT64_C(63) - (uint64_t)__builtin_clzll( bitset );
    bitset = bitset & ~(UINT64_C(1) << offset);
    idx -= 1;
  }

  uint64_t offset = UINT64_C(63) - (uint64_t)__builtin_clzll( bitset );
  return UINT64_C(1) << offset;
}
