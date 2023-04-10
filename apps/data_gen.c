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

#include "../datastruct/game_set.h"
#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libothello/othello.h"
#include "../misc/wthor.h"

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

#define N_TRIALS 5000

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
    for( size_t move_idx=0;; ++move_idx ) {
      othello_move_ctx_t ctx[1];
      if( !othello_game_start_move( game, ctx, &winner ) ) break;

      if( move_idx>=60 ) {
        /* Some of the games in the file seem to not have finished by the end of
           the fixed 60 move allotment. They do not have a winner as they are
           not complete? */
        break;
      }

      /* Dispatch callback on the valid game. NOTE: we are not including the
         final game board when game is over */

      on_valid_game( cb_ctx, game );

      uint8_t move_byte = fgame->moves[move_idx];

      /* Some files seem to have an explict pass byte? */
      if( move_byte == 0 ) {
        bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
        if( !valid ) Fail( "tried to make invalid move" );
        continue; // continue to next byte
      }

      /* But not all of them.. If the known current player cannot make any
         moves, I guess they must have passed?

         Kinda irritating that we have to compute this move mask on every move
         so many times; maybe should restructure this? */

      if( ctx->n_own_moves == 0 ) {
        bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
        if( !valid ) Fail( "tried to make invalid move" );
        /* do not continue, the current move applies to next player */
      }

      /* reset move ctx since we switched players */
      if( !othello_game_start_move( game, ctx, &winner ) ) Fail( "game should not be over" );

      uint8_t x, y;
      decode_move( move_byte, &x, &y );

      bool valid = othello_game_make_move( game, ctx, othello_bit_mask( x, y ) );
      if( !valid ) {
        othello_board_print( game );
        Fail( "tried to make invalid move" );
      }
    }
  }
}

typedef struct {
  size_t       n_input_boards;
  game_set_t * game_set;
} ctx_t;

static void
add_game_to_set( void *                 _ctx,
                 othello_game_t const * game )
{
  ctx_t *      ctx      = _ctx;
  game_set_t * game_set = ctx->game_set;

  ctx->n_input_boards += 1;

  /* Computer plays as white, don't save this board */
  if( game->curr_player != OTHELLO_BIT_WHITE ) return;

  /* If the game is over, skip */

  othello_game_t const * ret = game_set_get( game_set, game, true );
  if( !ret ) Fail( "Ran out of space in hash set" );
}

int main( void )
{
  /* Estimate the max possible number of unique boards and estimated number of
     trials we will run. */

  size_t total_games = 0;
  for( size_t file_idx = 0; file_idx < ARRAY_SIZE( ALL_FILES ); ++file_idx ) {
    wthor_file_t const * file = mmap_file( ALL_FILES[file_idx] );
    total_games += wthor_file_n_games( file );
    munmap( (void*)file, wthor_file_mem_size( file ) );
  }

  printf( "There are %zu total games in the database.\n", total_games );

  /* Estimated max number of possible boards is total games found * 60 b.c. each
     game has up to 60 moves in the file format */

  size_t total_boards = 60*total_games;

  /* Create hash set large enough to store the theoretical number of boards.
     We're using the difference between the theoretical max and the actual value
     as our load factor. Also, the hash set rounds up to nearest power of two,
     so we'll have plenty of free space. */

  size_t       hash_set_alloc_sz = game_set_size( total_boards );
  void *       _mem              = malloc( hash_set_alloc_sz );
  game_set_t * game_set          = game_set_new( _mem, total_boards );

  /* Build out the set of all boards */
  ctx_t ctx[1];
  ctx->n_input_boards = 0;
  ctx->game_set       = game_set;

  uint64_t st = wallclock();
  for( size_t file_idx = 0; file_idx < ARRAY_SIZE( ALL_FILES ); ++file_idx ) {
    wthor_file_t const * file = mmap_file( ALL_FILES[file_idx] );
    run_all_games_in_file( file, ctx, add_game_to_set );
    munmap( (void*)file, wthor_file_mem_size( file ) );
  }
  uint64_t ed = wallclock();

  size_t n_boards        = ctx->n_input_boards;
  size_t n_lookups       = game_set->n_gets; // each board results in one lookup
  size_t n_unique_boards = game_set_n_occupied( game_set );

  /* Crude because mmap IO is included here */
  double sec            = (double)(ed-st)/1e9;
  double lookup_per_sec = (double)n_lookups/sec;

  printf( "Found %zu unique boards for WHITE player (of %zu total) in %0.3f seconds. Ran ~%0.3f lookups per second\n",
          n_unique_boards, n_boards, sec, lookup_per_sec );

  /* For each game in the game set, run a bunch of random trials and generate
     some training data */

  size_t trials_run = 0;
  size_t total_trials = n_unique_boards*N_TRIALS;

  FILE * board_file = fopen( "training.boards", "w" );
  if( !board_file ) Fail( "Failed to open boards file" );

  FILE * pred_file = fopen( "training.pred", "w" );
  if( !pred_file ) Fail( "Failed to open pred file" );

  st = wallclock();
  #pragma omp parallel for
  for( size_t i = 0; i < game_set->n_slots; ++i ) {
    othello_game_t const * slot_game = &game_set->slots[i];
    if( slot_game->curr_player == OTHELLO_GAME_SET_SENTINEL ) continue;

    double wins = 0;

    for( size_t trial = 0; trial < N_TRIALS; ++trial ) {
      othello_game_t game[1] = { *slot_game };

      uint8_t winner = othello_game_random_playout( game, hash_u64( trial ) );
      if( winner==OTHELLO_BIT_WHITE ) wins += 1.0;
      if( winner==OTHELLO_GAME_TIED ) wins += 0.5;

      trials_run += 1;

      #pragma omp critical
      if( trials_run%100000 == 0 ) {
        uint64_t now            = wallclock();
        double   sec            = (double)(now-st)/1e9;
        double   trials_per_sec = (double)trials_run/sec;
        double   sec_remain     = (double)(total_trials-trials_run)/trials_per_sec;

        printf( "On trial %zu/%zu (%0.3f %%). Running %0.3f trials/sec. Est %0.3f min remain\n",
                trials_run, total_trials, (double)trials_run/(double)total_trials * 100.0,
                trials_per_sec, sec_remain/60.0 );
      }
    }

    /* input to NN is a 128 element flattened out floating point array */

    float  board_out[128] = { 0 };
    size_t idx = 0;

    for( size_t y = 0; y < 8; ++y ) {
      for( size_t x = 0; x < 8; ++x ) {
        bool occupied = slot_game->white & othello_bit_mask( x, y );
        board_out[idx++] = occupied ? 1.0f : 0.0f;
      }
    }

    for( size_t y = 0; y < 8; ++y ) {
      for( size_t x = 0; x < 8; ++x ) {
        bool occupied = slot_game->black & othello_bit_mask( x, y );
        board_out[idx++] = occupied ? 1.0f : 0.0f;
      }
    }

    assert( idx == 128 );

    #pragma omp critical
    {
      if( 1!=fwrite( board_out, sizeof(board_out), 1, board_file ) ) {
        Fail( "Failed to write to output file" );
      }

      float pred = (float)( (float)wins/(float)N_TRIALS );

      if( 1!=fwrite( &pred, sizeof(pred), 1, pred_file ) ) {
        Fail( "Failed to write to output file" );
      }
    }

    //if( trials_run > 2000000 ) break;
  }

  fclose( board_file );
  fclose( pred_file );

  free( (void*)game_set );
}
