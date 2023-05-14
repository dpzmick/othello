/* Train NN to estimate professioal moves */

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

static char const * const ALL_FILES[] = {
  "wthor_files/WTH_1977.wtb", "wthor_files/WTH_1978.wtb", "wthor_files/WTH_1979.wtb",
  "wthor_files/WTH_1980.wtb", "wthor_files/WTH_1981.wtb", "wthor_files/WTH_1982.wtb",
  "wthor_files/WTH_1983.wtb", "wthor_files/WTH_1984.wtb", "wthor_files/WTH_1985.wtb",
  "wthor_files/WTH_1986.wtb", "wthor_files/WTH_1987.wtb", "wthor_files/WTH_1988.wtb",
  "wthor_files/WTH_1989.wtb", "wthor_files/WTH_1990.wtb", "wthor_files/WTH_1991.wtb",
  "wthor_files/WTH_1992.wtb", "wthor_files/WTH_1993.wtb", "wthor_files/WTH_1994.wtb",
  "wthor_files/WTH_1995.wtb", "wthor_files/WTH_1996.wtb", "wthor_files/WTH_1997.wtb",
  "wthor_files/WTH_1998.wtb", "wthor_files/WTH_1999.wtb", "wthor_files/WTH_2000.wtb",
  "wthor_files/WTH_2001.wtb", "wthor_files/WTH_2002.wtb", "wthor_files/WTH_2003.wtb",
  "wthor_files/WTH_2004.wtb", "wthor_files/WTH_2005.wtb", "wthor_files/WTH_2006.wtb",
  "wthor_files/WTH_2007.wtb", "wthor_files/WTH_2008.wtb", "wthor_files/WTH_2009.wtb",
  "wthor_files/WTH_2010.wtb", "wthor_files/WTH_2011.wtb", "wthor_files/WTH_2012.wtb",
  "wthor_files/WTH_2013.wtb", "wthor_files/WTH_2014.wtb", "wthor_files/WTH_2015.wtb",
  "wthor_files/WTH_2016.wtb", "wthor_files/WTH_2017.wtb", "wthor_files/WTH_2018.wtb",
  "wthor_files/WTH_2019.wtb", "wthor_files/WTH_2020.wtb", "wthor_files/WTH_2021.wtb",
  "wthor_files/WTH_2022.wtb",
};

static wthor_file_t const *
mmap_file( char const * fname )
{
  int fd = open( fname, O_RDONLY );

  wthor_header_t hdr[1];
  if( sizeof(hdr)!=read( fd, hdr, sizeof(hdr) ) ) {
    Fail( "Failed to read header from file %s", strerror( errno ) );
  }

  /* 0|8 -> 8x8
     10 -> 10x10 */
  if( hdr->p1 != 0 && hdr->p1 != 8 ) {
    Fail( "Unexpected value in header" );
  }

  size_t sz = sizeof( wthor_game_t ) * hdr->n1;
  void * mem = mmap( NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0 );
  if( mem==MAP_FAILED ) {
    Fail( "Failed to mmap with %s", strerror( errno ) );
  }

  close( fd );
  return mem;
}

static void
format_nn_game_input( float *                    ret,
                      othello_game_t const *     game,
                      othello_move_ctx_t const * ctx )
{
  /* save to the input vector:
     1. the current player
     2. the valid moves (64)
     3. the board */

  size_t idx = 0;

  ret[idx++] = (float)game->curr_player;

  // ret[1 + x + y*8] = can_play
  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      ret[idx++] = ctx->own_moves & othello_bit_mask( x, y ) ? 1.0f : 0.0f;
    }
  }

  // save board, also row major
  // each player in separate array
  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      bool occupied = game->white & othello_bit_mask( x, y );
      ret[idx++] = occupied ? 1.0f : 0.0f;
    }
  }

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      bool occupied = game->black & othello_bit_mask( x, y );
      ret[idx++] = occupied ? 1.0f : 0.0f;
    }
  }

  assert( idx == 1+64+128 );
}

static void
flip_180( uint64_t   x,
          uint64_t   y,
          uint64_t * out_x,
          uint64_t * out_y )
{
  // 0 -> 7
  // 1 -> 6
  // ..
  // 7 -> 0

  *out_x = 7 - x;
  *out_y = 7 - y;
}


// flip 180 degress
// FIXME find more flips?
static othello_game_t
flip_game( othello_game_t const * game )
{
  othello_game_t ret[1];
  memset( ret, 0, sizeof(ret) );

  ret->curr_player = game->curr_player;

  for( uint64_t x = 0; x < 8; ++x ) {
    for( uint64_t y = 0; y < 8; ++y ) {
      uint64_t flipx, flipy;
      flip_180( x, y, &flipx, &flipy );

      if( game->white & othello_bit_mask( x, y ) ) {
        // should be no piece here yet in new board
        assert( (ret->white & othello_bit_mask( flipx, flipy )) ==0 );
        assert( (ret->black & othello_bit_mask( flipx, flipy )) ==0 );

        ret->white |= othello_bit_mask( flipx, flipy );
      }

      if( game->black & othello_bit_mask( x, y ) ) {
        // should be no piece here yet in new board
        assert( (ret->white & othello_bit_mask( flipx, flipy )) ==0 );
        assert( (ret->black & othello_bit_mask( flipx, flipy )) ==0 );

        ret->black |= othello_bit_mask( flipx, flipy );
      }
    }
  }

  return *ret;
}

/* static game_set_t * G_set = NULL; */

static void
save_game( uint64_t                   id,
           othello_game_t const *     game,
           othello_move_ctx_t const * ctx,
           uint8_t                    move_x,
           uint8_t                    move_y,
           FILE *                     game_ids,
           FILE *                     input_file,
           FILE *                     policy_file)
{
  float input[193] = { 0 };
  float policy[64] = { 0 };

  /* if( !G_set ) { */
  /*   size_t n = 15693850; */
  /*   size_t sz = game_set_size( n ); */

  /*   void * mem; */
  /*   if( 0!=posix_memalign( &mem, 4096, sz ) ) Fail( "allocation" ); */

  /*   memset( mem, 0, sz ); */

  /*   G_set = game_set_new( mem, n ); */
  /*   if( !G_set ) Fail( "game set" ); */
  /* } */

  /* If we've already seen a board, don't use it again. Just assume first expert
     was the best! FIXME removed as probably not a good idea to do it this way */

  /* if( game_set_get( G_set, game, false ) ) return; // already have it */
  /* if( !game_set_get( G_set, game, true ) ) Fail( "out of space" ); */

  // ----------------

  format_nn_game_input( input, game, ctx );

  if( 1!=fwrite( &id, sizeof(id), 1, game_ids ) ) {
    Fail( "Failed to write to game ids file" );
  }

  if( 1!=fwrite( input, sizeof(input), 1, input_file ) ) {
    Fail( "Failed to write to output file" );
  }

  if( move_x != 9 ) { // HACK
    policy[move_x + move_y*8] = 1.0;
  }

  if( 1!=fwrite( policy, sizeof(policy), 1, policy_file ) ) {
    Fail( "Failed to write to output file" );
  }

  // ----------------

  uint8_t            winner;
  othello_move_ctx_t flipped_ctx[1];
  othello_game_t     flipped_game[1] = { flip_game( game ) };

  if( !othello_game_start_move( flipped_game, flipped_ctx, &winner ) ) {
    othello_board_print( game );
    othello_board_print( flipped_game );
    Fail( "failed to start flip game move" );
  }

  format_nn_game_input( input, flipped_game, flipped_ctx );

  if( 1!=fwrite( &id, sizeof(id), 1, game_ids ) ) {
    Fail( "Failed to write to game ids file" );
  }

  if( 1!=fwrite( input, sizeof(input), 1, input_file ) ) {
    Fail( "Failed to write to output file" );
  }

  if( move_x != 9 ) { // HACK
    uint64_t flip_x, flip_y;
    flip_180( move_x, move_y, &flip_x, &flip_y );
    policy[flip_x + flip_y*8] = 1.0;
  }

  if( 1!=fwrite( policy, sizeof(policy), 1, policy_file ) ) {
    Fail( "Failed to write to output file" );
  }
}

static uint64_t
run_all_games_in_file( wthor_file_t const * file,
                       uint64_t             starting_id,
                       FILE *               game_ids,
                       FILE *               input_file,
                       FILE *               policy_file )
{
  for( size_t game_idx = 0; game_idx < wthor_file_n_games( file ); ++game_idx ) {
    wthor_game_t const * fgame = file->games + game_idx;

    /* Run the game, saving each board state and the move to take. */

    othello_game_t game[1];
    othello_game_init( game );

    /* Run the game, saving each state _and_ the move that was taken */

    uint8_t winner = (uint8_t)-1;
    for( size_t move_idx=0;; ++move_idx ) {
      othello_move_ctx_t ctx[1];
      if( !othello_game_start_move( game, ctx, &winner ) ) break;

      if( move_idx>=60 ) {
        /* Some of the games in the file seem to not have finished by the end of
           the fixed 60 move allotment. They do not have a winner as they are
           not complete? */
        break;
      }

      uint8_t move_byte  = fgame->moves[move_idx];

      uint8_t x, y;
      decode_move( move_byte, &x, &y );

      /* Some files have explicit pass byte */

      if( move_byte == 0 ) {
        save_game( starting_id, game, ctx, 9, 9, game_ids, input_file, policy_file );

        bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
        if( !valid ) Fail( "tried to make invalid move" );
        continue;
      }

      /* But not all of them.. If the known current player cannot make any
         moves, I guess they must have passed?

         Kinda irritating that we have to compute this move mask on every move
         so many times; maybe should restructure this? */

      if( ctx->n_own_moves == 0 ) {
        save_game( starting_id, game, ctx, 9, 9, game_ids, input_file, policy_file );

        bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
        if( !valid ) Fail( "tried to make invalid move" );
        /* do not continue, the current move applies to next player */
      }

      /* reset move ctx since we may have switched players */
      if( !othello_game_start_move( game, ctx, &winner ) ) Fail( "game should not be over" );

      save_game( starting_id, game, ctx, x, y, game_ids, input_file, policy_file );

      bool valid = othello_game_make_move( game, ctx, othello_bit_mask( x, y ) );
      if( !valid ) {
        othello_board_print( game );
        Fail( "player %d tried to make invalid move at (%d, %d)", game->curr_player, x, y );
      }
    }

    starting_id += 1;
  }

  return starting_id;
}

int main( void )
{
  FILE * ids_file = fopen( "sym_ids.dat", "w" );
  if( !ids_file ) Fail( "Failed to open board ids file" );

  FILE * input_file = fopen( "sym_input.dat", "w" );
  if( !input_file ) Fail( "Failed to open input file" );

  FILE * policy_file = fopen( "sym_policy.dat", "w" );
  if( !policy_file ) Fail( "Failed to open policy file" );

  uint64_t id = 0;
  for( size_t file_idx = 0; file_idx < ARRAY_SIZE( ALL_FILES ); ++file_idx ) {
    printf( "On file %zu of %zu\n", file_idx, ARRAY_SIZE( ALL_FILES ) );
    wthor_file_t const * file = mmap_file( ALL_FILES[file_idx] );

    id = run_all_games_in_file( file, id, ids_file, input_file, policy_file );
  }

  fclose( policy_file );
  fclose( input_file );
  fclose( ids_file );

  return 0;
}
