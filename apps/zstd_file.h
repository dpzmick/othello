#pragma once

#include "../libcommon/common.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zstd.h>

typedef struct {
  ZSTD_CStream * c;

  // we have to buffer our own IO so use unbuffered fd
  // and our own buffer
  char   buffer[4094];
  size_t buffer_pos;
  int    fd;
} zstd_file_t;


zstd_file_t *
zstd_file_writer( char const * fname )
{
  zstd_file_t * file = malloc( sizeof(*file) );
  memset( file, 0, sizeof(*file) );

  file->c = ZSTD_createCStream();
  if( !file->c ) Fail( "Failed to create compression stream" );

  file->fd = open( fname, O_CREAT | O_WRONLY | O_TRUNC, 0660 );
  if( file->fd == -1 ) Fail( "Failed to create output file %s with %s", fname, strerror( errno ) );

  return file;
}

void
zstd_file_writer_close( zstd_file_t * wtr )
{
  while( 1 ) {
    // do we need to flush output buffer?
    if( wtr->buffer_pos == sizeof(wtr->buffer) ) {
      ssize_t n = write( wtr->fd, wtr->buffer, sizeof(wtr->buffer) );
      if( (size_t)n != sizeof(wtr->buffer) ) Fail( "Failed to write to output file" );

      // printf( "wrote %zu to output file\n", n );

      wtr->buffer_pos = 0;
    }

    /* do the zstd dance, consuming any input it has for us and writing as much as possible to our buffer */

    ZSTD_outBuffer out = {
      .dst = wtr->buffer + wtr->buffer_pos,
      .size = sizeof(wtr->buffer) - wtr->buffer_pos,
      .pos = 0, /* always start at 0 */
    };


    size_t ret = ZSTD_endStream( wtr->c, &out );
    if( ZSTD_isError( ret ) ) Fail( "Failed to compress" );
    wtr->buffer_pos += out.pos; // this much was written to our buffer

    if( ret == 0 ) {
      // flush anything else left in the buffer

      ssize_t n = write( wtr->fd, wtr->buffer, wtr->buffer_pos );
      if( (size_t)n != wtr->buffer_pos ) Fail( "Failed to write to output file" );

      break; // we're done!
    }
  }

  if( 0 != close( wtr->fd ) ) Fail( "Failed to close file" );
  ZSTD_freeCStream( wtr->c );
  free( wtr );
}

void
zstd_file_writer_write( zstd_file_t * wtr,
                        char const *  buf,
                        size_t        len )
{
  while( len ) {
    // do we need to flush output buffer?
    if( wtr->buffer_pos == sizeof(wtr->buffer) ) {
      ssize_t n = write( wtr->fd, wtr->buffer, sizeof(wtr->buffer) );
      if( (size_t)n != sizeof(wtr->buffer) ) Fail( "Failed to write to output file" );

      wtr->buffer_pos = 0;
    }

    /* do the zstd dance, consuming any input it has for us and writing as much as possible to our buffer */

    ZSTD_outBuffer out = {
      .dst = wtr->buffer + wtr->buffer_pos,
      .size = sizeof(wtr->buffer) - wtr->buffer_pos,
      .pos = 0, /* always start at 0 */
    };

    ZSTD_inBuffer in = {
      .src = buf,
      .size = len,
      .pos = 0,
    };

    size_t ret = ZSTD_compressStream( wtr->c, &out, &in );
    if( ZSTD_isError( ret ) ) Fail( "Failed to compress" );

    wtr->buffer_pos += out.pos; // this much was written to our buffer
    len             -= in.pos;  // this much was consumed from zstd
  }
}
