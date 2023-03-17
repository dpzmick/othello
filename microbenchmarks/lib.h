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

void
fast_copy( uint8_t const * restrict a,
           uint8_t * restrict       b,
           size_t                   n );

void
stream_copy( float const * restrict a,
             float * restrict       b,
             size_t                 n );

void
stream_copy2( float const * restrict a,
              float * restrict       b,
              size_t                 n );

void
stream_scale( float const * restrict a,
              float * restrict       b,
              float                  q,
              size_t                 n );

void
stream_scale2( float const * restrict a,
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
