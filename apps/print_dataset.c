#include "../datastruct/game_set.h"
#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libcomputer/mcts.h"
#include "../libothello/othello.h"
#include "../misc/wthor.h"

#include <fcntl.h>
#include <inttypes.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <unistd.h>

int main( void )
{
  /* FILE * ids_file = fopen( "sym_ids_playernorm.dat", "r" ); */
  /* if( !ids_file ) Fail( "Failed to open board ids file" ); */

  /* FILE * input_file = fopen( "sym_input_playernorm.dat", "r" ); */
  /* if( !input_file ) Fail( "Failed to open input file" ); */

  /* FILE * policy_file = fopen( "sym_policy_playernorm.dat", "r" ); */
  /* if( !policy_file ) Fail( "Failed to open policy file" ); */
  FILE * ids_file = fopen( "sym_ids.dat", "r" );
  if( !ids_file ) Fail( "Failed to open board ids file" );

  FILE * input_file = fopen( "sym_input.dat", "r" );
  if( !input_file ) Fail( "Failed to open input file" );

  FILE * policy_file = fopen( "sym_policy.dat", "r" );
  if( !policy_file ) Fail( "Failed to open policy file" );

  while( 1 ) {
    uint64_t id;
    if( 1!=fread( &id, sizeof(id), 1, ids_file ) ) break;

    float input[193];
    /* float input[192]; */
    if( 1!=fread( &input, sizeof(input), 1, input_file ) ) Fail( "read input" );

    float policy[64];
    if( 1!=fread( &policy, sizeof(policy), 1, policy_file ) ) Fail( "read policy" );

    /* printf( "player: %f\n", (double)input[0] ); */

    /* printf( "valid: "); */
    /* for( size_t i = 0; i < 64; ++i ) { */
    /*   printf( "%f, ",  (double)input[1 + i] ); */
    /* } */
    /* printf( "\n" ); */

    /* printf( "white: "); */
    /* for( size_t i = 0; i < 64; ++i ) { */
    /*   printf( "%f, ",  (double)input[1 + 64 + i] ); */
    /* } */
    /* printf( "\n" ); */

    /* printf( "black: "); */
    /* for( size_t i = 0; i < 64; ++i ) { */
    /*   printf( "%f, ",  (double)input[1 + 64 + 64 + i] ); */
    /* } */
    /* printf( "\n" ); */

    // display board, overlaying valid moves and the selected policy we are training?

    printf("  | 0 ");
    for( size_t x = 1; x < 8; ++x ) {
      printf("%zu ", x);
    }
    printf("\n--+----------------\n");

    for( size_t y = 0; y < 8; ++y ) {
      printf("%zu | ", y );

      for( size_t x = 0; x < 8; ++x ) {
        uint8_t bit_valid  = input[1 + x + y*8] > 0.0f;
        uint8_t bit_policy = policy[x + y*8] > 0.0f;
        uint8_t bit_white  = input[1 + 64 + x + y*8] > 0.0f;
        uint8_t bit_black  = input[1 + 64 + 64 + x + y*8] > 0.0f;

        /* uint8_t bit_valid  = input[x + y*8] > 0.0f; */
        /* uint8_t bit_policy = policy[x + y*8] > 0.0f; */
        /* uint8_t bit_white  = input[64 + x + y*8] > 0.0f; */
        /* uint8_t bit_black  = input[64 + 64 + x + y*8] > 0.0f; */

        if( bit_valid ) {
          assert( !bit_white && !bit_black );
          if( bit_policy ) {
            printf( "# " );
          } else {
            printf( ". " );
          }
        }
        else if( bit_white ) {
          printf( "W " );
        }
        else if( bit_black ) {
          printf( "B " );
        }
        else {
          printf( "  " );
        }
      }
      printf("\n");
    }
  }

  fclose( policy_file );
  fclose( input_file );
  fclose( ids_file );

  return 0;
}
