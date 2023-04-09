#pragma once

#include <stdint.h>

/* mumur3 hash finalizer */

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

/* A cleaner implementation of xxhash-r39 (Open Source BSD licensed). Borrowed
   from firedancer */

#define ROTATE_LEFT(x,r) (((x)<<(r)) | ((x)>>(64-(r))))
#define C1 (11400714785074694791UL)
#define C2 (14029467366897019727UL)
#define C3 ( 1609587929392839161UL)
#define C4 ( 9650029242287828579UL)
#define C5 ( 2870177450012600261UL)

static inline uint64_t
fd_hash( uint64_t     seed,
         void const * buf,
         uint64_t     sz ) {
  unsigned char const * p    = ((unsigned char const *)buf);
  unsigned char const * stop = p + sz;

  uint64_t h;

  if( sz<32 ) h = seed + C5;
  else {
    unsigned char const * stop32 = stop - 32;
    uint64_t              w      = seed + (C1+C2);
    uint64_t              x      = seed + C2;
    uint64_t              y      = seed;
    uint64_t              z      = seed - C1;

    do { /* All complete blocks of 32 */
      w += (((uint64_t const *)p)[0])*C2; w = ROTATE_LEFT( w, 31 ); w *= C1;
      x += (((uint64_t const *)p)[1])*C2; x = ROTATE_LEFT( x, 31 ); x *= C1;
      y += (((uint64_t const *)p)[2])*C2; y = ROTATE_LEFT( y, 31 ); y *= C1;
      z += (((uint64_t const *)p)[3])*C2; z = ROTATE_LEFT( z, 31 ); z *= C1;
      p += 32;
    } while( p<=stop32 );

    h = ROTATE_LEFT( w, 1 ) + ROTATE_LEFT( x, 7 ) + ROTATE_LEFT( y, 12 ) + ROTATE_LEFT( z, 18 );

    w *= C2; w = ROTATE_LEFT( w, 31 ); w *= C1; h ^= w; h = h*C1 + C4;
    x *= C2; x = ROTATE_LEFT( x, 31 ); x *= C1; h ^= x; h = h*C1 + C4;
    y *= C2; y = ROTATE_LEFT( y, 31 ); y *= C1; h ^= y; h = h*C1 + C4;
    z *= C2; z = ROTATE_LEFT( z, 31 ); z *= C1; h ^= z; h = h*C1 + C4;
  }

  h += ((uint64_t)sz);

  while( (p+8)<=stop ) { /* Last 1 to 3 complete uint64_t's */
    uint64_t w = ((uint64_t const *)p)[0];
    w *= C2; w = ROTATE_LEFT( w, 31 ); w *= C1; h ^= w; h = ROTATE_LEFT( h, 27 )*C1 + C4;
    p += 8;
  }

  if( (p+4)<=stop ) { /* Last complete uint */
    uint64_t w = ((uint64_t)(((uint32_t const *)p)[0]));
    w *= C1; h ^= w; h = ROTATE_LEFT( h, 23 )*C2 + C3;
    p += 4;
  }

  while( p<stop ) { /* Last 1 to 3 unsigned char's */
    uint64_t w = ((uint64_t)(p[0]));
    w *= C5; h ^= w; h = ROTATE_LEFT( h, 11 )*C1;
    p++;
  }

  /* Final avalanche */
  h ^= h >> 33;
  h *= C2;
  h ^= h >> 29;
  h *= C3;
  h ^= h >> 32;

  return h;
}

/* uint64_t */
/* fd_hash_memcpy( uint64_t                    seed, */
/*                 void *       FD_RESTRICT dst, */
/*                 void const * FD_RESTRICT src, */
/*                 uint64_t                    sz ) { */
/*   unsigned char       * FD_RESTRICT q    = ((unsigned char       *)dst); */
/*   unsigned char const * FD_RESTRICT p    = ((unsigned char const *)src); */
/*   unsigned char const * FD_RESTRICT stop = p + sz; */

/*   uint64_t h; */

/*   if( sz<32 ) h = seed + C5; */
/*   else { */
/*     unsigned char const * FD_RESTRICT stop32 = stop - 32; */
/*     uint64_t w = seed + (C1+C2); */
/*     uint64_t x = seed + C2; */
/*     uint64_t y = seed; */
/*     uint64_t z = seed - C1; */

/*     do { /\* All complete blocks of 32 *\/ */
/*       uint64_t p0 = ((uint64_t const *)p)[0]; */
/*       uint64_t p1 = ((uint64_t const *)p)[1]; */
/*       uint64_t p2 = ((uint64_t const *)p)[2]; */
/*       uint64_t p3 = ((uint64_t const *)p)[3]; */
/*       w += p0*C2; w = ROTATE_LEFT( w, 31 ); w *= C1; */
/*       x += p1*C2; x = ROTATE_LEFT( x, 31 ); x *= C1; */
/*       y += p2*C2; y = ROTATE_LEFT( y, 31 ); y *= C1; */
/*       z += p3*C2; z = ROTATE_LEFT( z, 31 ); z *= C1; */
/*       ((uint64_t *)q)[0] = p0; */
/*       ((uint64_t *)q)[1] = p1; */
/*       ((uint64_t *)q)[2] = p2; */
/*       ((uint64_t *)q)[3] = p3; */
/*       p += 32; */
/*       q += 32; */
/*     } while( p<=stop32 ); */

/*     h = ROTATE_LEFT( w, 1 ) + ROTATE_LEFT( x, 7 ) + ROTATE_LEFT( y, 12 ) + ROTATE_LEFT( z, 18 ); */

/*     w *= C2; w = ROTATE_LEFT( w, 31 ); w *= C1; h ^= w; h = h*C1 + C4; */
/*     x *= C2; x = ROTATE_LEFT( x, 31 ); x *= C1; h ^= x; h = h*C1 + C4; */
/*     y *= C2; y = ROTATE_LEFT( y, 31 ); y *= C1; h ^= y; h = h*C1 + C4; */
/*     z *= C2; z = ROTATE_LEFT( z, 31 ); z *= C1; h ^= z; h = h*C1 + C4; */
/*   } */

/*   h += ((uint64_t)sz); */

/*   while( (p+8)<=stop ) { /\* Last 1 to 3 complete uint64_t's *\/ */
/*     uint64_t p0 = ((uint64_t const *)p)[0]; */
/*     uint64_t w  = p0*C2; w = ROTATE_LEFT( w, 31 ); w *= C1; h ^= w; h = ROTATE_LEFT( h, 27 )*C1 + C4; */
/*     ((uint64_t *)q)[0] = p0; */
/*     p += 8; */
/*     q += 8; */
/*   } */

/*   if( (p+4)<=stop ) { /\* Last complete uint *\/ */
/*     uint32_t p0 = ((uint32_t const *)p)[0]; */
/*     uint64_t w = ((uint64_t)p0)*C1; h ^= w; h = ROTATE_LEFT( h, 23 )*C2 + C3; */
/*     ((uint *)q)[0] = p0; */
/*     p += 4; */
/*     q += 4; */
/*   } */

/*   while( p<stop ) { /\* Last 1 to 3 unsigned char's *\/ */
/*     unsigned char p0 = p[0]; */
/*     uint64_t w  = ((uint64_t)p0)*C5; h ^= w; h = ROTATE_LEFT( h, 11 )*C1; */
/*     q[0] = p0; */
/*     p++; */
/*     q++; */
/*   } */

/*   /\* Final avalanche *\/ */
/*   h ^= h >> 33; */
/*   h *= C2; */
/*   h ^= h >> 29; */
/*   h *= C3; */
/*   h ^= h >> 32; */

/*   return h; */
/* } */

#undef C5
#undef C4
#undef C3
#undef C2
#undef C1
#undef ROTATE_LEFT
