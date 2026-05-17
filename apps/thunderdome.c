/* Pitting algorithms against eachother */

#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libcomputer/mcts.h"
#include "../libcomputer/nn_policy.h"
#include "../libothello/othello.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t trials = 1000;
static const size_t mcts_trials_per_move = 2000;
static const size_t mcts_table_slots     = (1<<16);

#if 0
static uint64_t
select_random_move( uint64_t valid_moves,
                    uint64_t seed )
{
  uint64_t valid_moves_cnt  = (uint64_t)__builtin_popcountll( valid_moves );
  if( valid_moves_cnt==0 ) return OTHELLO_MOVE_PASS;

  uint64_t rand_move_idx = hash_u64( seed ) % valid_moves_cnt;
  return keep_ith_set_bit( valid_moves, rand_move_idx );
}
#endif

/* Wilson 95% CI half-width on a winrate of k/n. */
static void
wilson_ci_95( size_t k, size_t n, double * lo, double * hi )
{
  double p = (double)k / (double)n;
  double z = 1.96;
  double z2 = z * z;
  double denom = 1.0 + z2 / (double)n;
  double center = (p + z2 / (2.0 * (double)n)) / denom;
  double half = z * sqrt((p*(1.0-p) + z2/(4.0*(double)n)) / (double)n) / denom;
  *lo = center - half;
  *hi = center + half;
}

int
main( void )
{
  mcts_state_t * mcts_state = malloc( mcts_state_size( mcts_table_slots ) );
  if( !mcts_state ) Fail( "failed to allocate" );

  size_t nn_wins   = 0;
  size_t mcts_wins = 0;
  size_t draws     = 0;

  /* Track wins separately by which color the NN played, so we can spot
     color-asymmetric strength (the NN might be much better as white if it
     was trained mostly on white-to-move positions, etc -- canonicalization
     should fix that but worth verifying). */
  size_t nn_wins_as_white = 0;
  size_t nn_games_as_white = 0;
  size_t nn_wins_as_black = 0;
  size_t nn_games_as_black = 0;

  othello_game_t game[1];
  for( size_t trial = 0; trial < trials; ++trial ) {
    /* Alternate which color the NN plays. Halves variance from one color
       being intrinsically harder against this MCTS. */
    bool    nn_is_white = (trial % 2) == 0;
    uint8_t mcts_color  = nn_is_white ? OTHELLO_BIT_BLACK : OTHELLO_BIT_WHITE;

    othello_game_init( game );
    mcts_state_init( mcts_state, mcts_trials_per_move, mcts_color,
                     hash_u64( (uint64_t)trial ), mcts_table_slots );

    uint8_t winner;
    while( 1 ) {
      othello_move_ctx_t ctx[1];
      uint64_t           move;
      bool               valid;

      if( !othello_game_start_move( game, ctx, &winner ) ) break;

      if( game->curr_player == mcts_color ) {
        move = mcts_select_move( mcts_state, game, ctx );
      }
      else {
        move = nn_policy_select_move( game, ctx );
      }

      valid = othello_game_make_move( game, ctx, move );
      if( !valid ) Fail( "move invalid" );
    }

    /* Tally for the NN's perspective regardless of color. */
    if( winner == OTHELLO_GAME_TIED ) {
      draws += 1;
    }
    else if( (winner == OTHELLO_BIT_WHITE) == nn_is_white ) {
      nn_wins += 1;
      if( nn_is_white ) nn_wins_as_white += 1; else nn_wins_as_black += 1;
    }
    else {
      mcts_wins += 1;
    }
    if( nn_is_white ) nn_games_as_white += 1; else nn_games_as_black += 1;
  }

  double winrate = (double)nn_wins / (double)trials;
  double lo, hi;
  wilson_ci_95( nn_wins, trials, &lo, &hi );

  printf( "NN: %zu  MCTS: %zu  Draws: %zu  (of %zu)\n",
          nn_wins, mcts_wins, draws, trials );
  printf( "NN winrate: %.3f  Wilson 95%% CI: [%.3f, %.3f]\n", winrate, lo, hi );
  printf( "  as white: %zu/%zu = %.3f\n",
          nn_wins_as_white, nn_games_as_white,
          (double)nn_wins_as_white / (double)nn_games_as_white );
  printf( "  as black: %zu/%zu = %.3f\n",
          nn_wins_as_black, nn_games_as_black,
          (double)nn_wins_as_black / (double)nn_games_as_black );

  free( mcts_state );
  return 0;
}
