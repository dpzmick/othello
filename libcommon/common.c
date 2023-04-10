#include "common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void
_fail( char const * file,
       int          line,
       char const * fmt, ... )
{
  va_list args;
  va_start( args, fmt );

  fprintf( stderr, "%s:%d: Fail: ", file, line );
  vfprintf( stderr, fmt, args );
  fprintf( stderr, "\n" );
  fflush( stderr );

  va_end( args );
  abort();
}
