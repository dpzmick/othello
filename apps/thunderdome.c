/* Pitting algorithms against eachother */

#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libcomputer/mcts.h"
#include "../libcomputer/nn.h"
#include "../libothello/othello.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t trials = 100;

/* static uint64_t */
/* select_random_move( uint64_t valid_moves, */
/*                     uint64_t seed ) */
/* { */
/*   uint64_t valid_moves_cnt  = (uint64_t)__builtin_popcountll( valid_moves ); */
/*   if( valid_moves_cnt==0 ) return OTHELLO_MOVE_PASS; */

/*   uint64_t rand_move_idx = hash_u64( seed ) % valid_moves_cnt; */
/*   return keep_ith_set_bit( valid_moves, rand_move_idx ); */
/* } */

int
main( void )
{
  mcts_state_t * black_player_state = malloc( mcts_state_size( 8192 ) );
  if( !black_player_state ) Fail( "failed to allocate" );

  mcts_state_t * white_player_state = malloc( mcts_state_size( 8192 ) );
  if( !white_player_state ) Fail( "failed to allocate" );

  size_t black_wins = 0;
  size_t white_wins = 0;
  size_t draws      = 0;

  othello_game_t game[1];
  for( size_t trial = 0; trial < trials; ++trial ) {
    othello_game_init( game );
    mcts_state_init( black_player_state, 64, OTHELLO_BIT_BLACK, hash_u64( (uint64_t)trial ), 8192 );
    mcts_state_init( white_player_state, 64, OTHELLO_BIT_WHITE, hash_u64( (uint64_t)trial ), 8192 );

    uint8_t winner;
    while( 1 ) {
      othello_move_ctx_t ctx[1];
      uint64_t           move;
      bool               valid;

      if( !othello_game_start_move( game, ctx, &winner ) ) break;

      move = mcts_select_move( black_player_state, game, ctx );
      valid = othello_game_make_move( game, ctx, move );
      if( !valid ) Fail( "move invalid" );

      if( !othello_game_start_move( game, ctx, &winner ) ) break;

      move = mcts_select_move( white_player_state, game, ctx );
      valid = othello_game_make_move( game, ctx, move );
      if( !valid ) Fail( "move invalid" );

      /* move = select_random_move( othello_game_all_valid_moves( game ), game->white*game->black+trial ); */
      /* valid = othello_game_make_move( game, move ); */
      /* if( !valid ) Fail( "move invalid" ); */

      /* move = nn_select_move( game ); */
      /* valid = othello_game_make_move( game, move ); */
      /* if( !valid ) Fail( "move invalid" ); */
    }

    if( winner==OTHELLO_BIT_BLACK ) black_wins += 1;
    if( winner==OTHELLO_BIT_WHITE ) white_wins += 1;
    if( winner==OTHELLO_GAME_TIED ) draws += 1;
  }

  printf( "Black won: %zu\nWhite won: %zu.\nDraws: %zu\n",
          black_wins, white_wins, draws );

  return 0;
}
