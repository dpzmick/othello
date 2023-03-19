#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct PlaydateAPI PlaydateAPI;

#define MAX( a, b ) ((a) < (b) ? (b) : (a))
#define ARRAY_SIZE( a ) (sizeof(a)/sizeof(a[0]))

static inline intptr_t
align_ptr_up( intptr_t ptr, intptr_t a )
{
  a--;
  return (ptr+a) & ~a;
}

void
stream_copy( float const * restrict a,
             float * restrict       b,
             size_t                 n );

void
stream_scale( float const * restrict a,
              float * restrict       b,
              float                  q,
              size_t                 n );

void
stream_sum( float const * restrict a,
            float const * restrict b,
            float * restrict       c,
            size_t                 n );

void
stream_triad( float const * restrict a,
              float const * restrict b,
              float * restrict       c,
              float                  q,
              size_t                 n );
