#pragma once

#include <stdint.h>
#include <time.h>

#define NS 1000000000UL

#define ARRAY_SIZE(a) (sizeof((a))/sizeof(*(a)))

void
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
