#pragma once

#include <stddef.h>
#include <stdint.h>

#define MAX( a, b ) ((a) < (b) ? (b) : (a))

static inline intptr_t
align_ptr_up( intptr_t ptr, intptr_t a )
{
  a--;
  return (ptr+a) & ~a;
}

// this isn't stricly the standard "stream" benchmark
// but the stdlib memcpy was beating a[i] = b[i]
// so trying to optimize a bit to better understand the
// architecture.
//
// this is tied with playdate stdlib
static void __attribute__((noinline))
fast_copy( uint8_t const * restrict a,
           uint8_t * restrict       b,
           size_t                   n )
{
  uint8_t const * ed = a+n;

  // peel off the front until we are aligned
  while( a < ed ) {
    // if we are aligned, break
    if( ((size_t)a & 0x20) == 0 ) break;

    // else copy as uint8_t
    *b++ = *a++;
  }

  // do main body as "word" sized loads and stores
  // this is going to run on 32bit arm so a word is u32
  while( a < ed && ed-a > 4 ) {
    uint32_t _a = *((uint32_t*)a);
    *((uint32_t*)b) = _a;

    a += 4;
    b += 4;
  }

  // finish up any funky sized tail
  while( a < ed ) {
    *b++ = *a++;
  }
}

// proper "stream" will run with floats and none of the fancy setup code.
// assumes that the buffer is aligned
static void __attribute__((noinline))
stream_copy( float  const * restrict a,
             float  * restrict       b,
             size_t                  n )
{
  for( size_t i = 0; i < n; ++i ) {
    b[i] = a[i];
  }
}
