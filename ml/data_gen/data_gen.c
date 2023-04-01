/* We are going to try using a neural net to evaluate each possible board
   position that could be played.

   To generate this data, we'll generate a large number of possible boards from
   expert games and perform a large number of random playouts from those boards.
   The random playouts aren't perfectly ideal, but this shouldn't be worse than
   the MCTS that I was doing before.

   Randomized playouts are rather expensive so we're using reasonably fast code
   to generate these, however this bitboard othello representiation was
   originally written for MCTS running on the device (which does not have vector
   unit) so it has only been tuned to the extent needed to get fast-enough for a
   first stab on the device. There's lots of potential being left on the floor
   in the game code.

   Keep in mind, the performance of the device we are targetting is extremely
   limited; we're not going to be able to run anything super sophisticated on
   the actual device. This sort of limits how sophisticated of training and
   input data we need (I think?). The net can't be very large, so there's not
   huge value in generating a giant corpus of example games (need a pretty big
   one), and it's not going to be able to predict very well, so we don't really
   need to fit it to super great data (I think?).

   FIXME consider/evaluate something similar to alpha-go self play here? Not
   sure that's valuable with a tiny net. */

#include "common.h"
#include "hash.h"
#include "othello.h"
#include "wthor.h"

#include <fcntl.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <unistd.h>

#define N_TRIALS 1000

static char const * const ALL_FILES[] = {
  "data/WTH_1977.wtb", "data/WTH_1978.wtb", "data/WTH_1979.wtb",
  "data/WTH_1980.wtb", "data/WTH_1981.wtb", "data/WTH_1982.wtb",
  "data/WTH_1983.wtb", "data/WTH_1984.wtb", "data/WTH_1985.wtb",
  "data/WTH_1986.wtb", "data/WTH_1987.wtb", "data/WTH_1988.wtb",
  "data/WTH_1989.wtb", "data/WTH_1990.wtb", "data/WTH_1991.wtb",
  "data/WTH_1992.wtb", "data/WTH_1993.wtb", "data/WTH_1994.wtb",
  "data/WTH_1995.wtb", "data/WTH_1996.wtb", "data/WTH_1997.wtb",
  "data/WTH_1998.wtb", "data/WTH_1999.wtb", "data/WTH_2000.wtb",
  "data/WTH_2001.wtb", "data/WTH_2002.wtb", "data/WTH_2003.wtb",
  "data/WTH_2004.wtb", "data/WTH_2005.wtb", "data/WTH_2006.wtb",
  "data/WTH_2007.wtb", "data/WTH_2008.wtb", "data/WTH_2009.wtb",
  "data/WTH_2010.wtb", "data/WTH_2011.wtb", "data/WTH_2012.wtb",
  "data/WTH_2013.wtb", "data/WTH_2014.wtb", "data/WTH_2015.wtb",
  "data/WTH_2016.wtb", "data/WTH_2017.wtb", "data/WTH_2018.wtb",
  "data/WTH_2019.wtb", "data/WTH_2020.wtb", "data/WTH_2021.wtb",
  "data/WTH_2022.wtb",
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

typedef void (*on_valid_game_t)( void *                 ctx,
                            othello_game_t const * game );

static void
run_all_games_in_file( wthor_file_t const * file,
                       void *               cb_ctx,
                       on_valid_game_t      on_valid_game )
{
  othello_game_t game[1];
  for( size_t game_idx = 0; game_idx < wthor_file_n_games( file ); ++game_idx ) {
    wthor_game_t const * fgame = file->games + game_idx;

    othello_game_init( game );

    // try to play out the game in the file
    uint8_t winner = (uint8_t)-1;
    for( size_t move_idx=0; !othello_game_is_over( game, &winner ); ++move_idx ) {
      if( move_idx>=60 ) {
        /* Some of the games in the file seem to not have finished by the end of
           the fixed 60 move allotment. They do not have a winner as they are
           not complete? */
        break;
      }

      /* Dispatch callback on the valid game */
      on_valid_game( cb_ctx, game );

      uint8_t move_byte = fgame->moves[move_idx];

      /* Some files seem to have an explict pass byte? */
      if( move_byte == 0 ) {
        bool valid = othello_game_make_move( game, OTHELLO_MOVE_PASS );
        assert( valid );
        continue; // continue to next byte
      }

      /* But not all of them.. If the known current player cannot make any
         moves, I guess they must have passed?

         Kinda irritating that we have to compute this move mask on every move
         so many times; maybe should restructure this? */

      uint64_t moves = othello_game_all_valid_moves( game );
      if( moves==0 ) {
        // printf( "[%zu.%zu] = PASS by %d\n", game_idx, move_idx, game->curr_player );
        bool valid = othello_game_make_move( game, OTHELLO_MOVE_PASS );
        assert( valid );
        // do not continue, the current move applies to next player
      }

      uint8_t x, y;
      decode_move( move_byte, &x, &y );

      // printf( "[%zu.%zu] = (%d,%d) by %d\n", game_idx, move_idx, x, y, game->curr_player );

      bool valid = othello_game_make_move( game, othello_bit_mask( x, y ) );
      if( !valid ) {
        othello_board_print( game );
        assert( false );
      }
    }

    /* printf( "[%zu] winner=%d\n", game_idx, winner ); */
    /* othello_board_print( game ); */
  }
}

typedef struct {
  FILE * board_file;
  FILE * pred_file;
  size_t trials_run;
} cb_ctx_t;

static void
on_valid_game( void *                 _ctx,
               othello_game_t const * _game )
{
  cb_ctx_t * ctx = _ctx;

  // computer plays are white
  if( _game->curr_player != OTHELLO_BIT_WHITE ) return;

  double wins = 0;

  for( size_t trial = 0; trial < N_TRIALS; ++trial ) {
    othello_game_t game[1] = { *_game };

    uint8_t winner = othello_game_random_playout( game, hash_u64( trial ) );
    if( winner==OTHELLO_BIT_WHITE ) wins += 1.0;
    if( winner==OTHELLO_GAME_TIED ) wins += 0.5;

    ctx->trials_run += 1;
  }

  // inputs are a 128bit flattened out floating point array

  float  board_out[128] = { 0 };
  size_t idx = 0;

  for( size_t x = 0; x < 8; ++x ) {
    for( size_t y = 0; y < 8; ++y ) {
      bool occupied = _game->white & othello_bit_mask( x, y );
      board_out[idx++] = occupied ? 1.0f : 0.0f;
    }
  }

  for( size_t x = 0; x < 8; ++x ) {
    for( size_t y = 0; y < 8; ++y ) {
      bool occupied = _game->black & othello_bit_mask( x, y );
      board_out[idx++] = occupied ? 1.0f : 0.0f;
    }
  }

  if( 1!=fwrite( board_out, sizeof(board_out), 1, ctx->board_file ) ) {
    Fail( "Failed to write to output file" );
  }

  float pred = (float)( (float)wins/(float)N_TRIALS );

  if( 1!=fwrite( &pred, sizeof(pred), 1, ctx->pred_file ) ) {
    Fail( "Failed to write to output file" );
  }
}

int main( void )
{
  size_t total_games = 0;

  for( size_t file_idx = 0; file_idx < ARRAY_SIZE( ALL_FILES ); ++file_idx ) {
    wthor_file_t const * file = mmap_file( ALL_FILES[file_idx] );
    total_games += wthor_file_n_games( file );
    munmap( (void*)file, wthor_file_mem_size( file ) );
  }

  size_t total_boards = 60*total_games;
  size_t total_evals  = N_TRIALS * total_boards;

  /* These are estimates */
  printf( "There are %zu total games in the database.\n", total_games );
  printf( "Resulting in %zu unique boards\n", total_boards );
  printf( "And an estimated %zu total trials\n", total_evals );
  printf( "----------------------------------------------\n" );

  /* The board data is about 2gigs raw, plus another little bit for the */

  for( size_t file_idx = 0; file_idx < ARRAY_SIZE( ALL_FILES ); ++file_idx ) {
    printf( "Working on file %zu of %zu\n", file_idx, ARRAY_SIZE( ALL_FILES ) );

    char buf[1024];
    char * bn = basename( (char*)ALL_FILES[file_idx] ); // hope it doesn't touch my buffer!

    cb_ctx_t ctx[1];
    ctx->trials_run = 0;

    sprintf( buf, "training/%s_boards", bn );
    ctx->board_file = fopen( buf, "w" );
    if( !ctx->board_file ) Fail( "board file %s", buf );

    sprintf( buf, "training/%s_pred", bn );
    ctx->pred_file = fopen( buf, "w" );
    if( !ctx->pred_file ) Fail( "pred file %s", buf );

    wthor_file_t const * file = mmap_file( ALL_FILES[file_idx] );
    uint64_t st = wallclock();
    run_all_games_in_file( file, ctx, on_valid_game );
    uint64_t ed = wallclock();

    total_evals = total_evals > ctx->trials_run ? total_evals - ctx->trials_run : 0;

    double sec            = (double)(ed-st)/1e9;
    double trial_per_sec  = (double)(ctx->trials_run)/sec;
    double time_to_finish = (double)total_evals/trial_per_sec;

    fclose( ctx->board_file );
    fclose( ctx->pred_file );

    printf( "... ran %f Games/Second. Total time remaining=%f hours\n", trial_per_sec, time_to_finish/60/60 );

    munmap( (void*)file, wthor_file_mem_size( file ) );
  }
}
