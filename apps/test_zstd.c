#include "zstd_file.h"

int
main( int     argc,
      char ** argv )
{
  if( argc != 3 ) Fail( "usage: %s input output\n", argv[0] );

  FILE * in = fopen( argv[1], "r" );
  if( !in ) Fail( "!in" );

  zstd_file_t * out = zstd_file_writer( argv[2] );

  while( 1 ) {
    char buffer[4046];
    size_t n = fread( buffer, 1, sizeof( buffer ), in );
    printf( "read %zu from input\n", n );

    zstd_file_writer_write( out, buffer, n );

    if( n != sizeof(buffer) ) break; // could have also been an error but it is a pain to check
  }

  zstd_file_writer_close( out );
}
