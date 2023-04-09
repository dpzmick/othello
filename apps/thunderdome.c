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
    mcts_state_init( black_player_state, 1000, OTHELLO_BIT_BLACK, hash_u64( (uint64_t)trial ), 8192 );
    mcts_state_init( white_player_state, 1000, OTHELLO_BIT_WHITE, hash_u64( (uint64_t)trial ), 8192 );

    uint8_t winner;
    while( !othello_game_is_over( game, &winner ) ) {
      uint64_t move;
      bool     valid;

      move = mcts_select_move( black_player_state, game );
      valid = othello_game_make_move( game, move );
      if( !valid ) Fail( "move invalid" );

      /* move = mcts_select_move( white_player_state, game ); */
      /* valid = othello_game_make_move( game, move ); */
      /* if( !valid ) Fail( "move invalid" ); */

      move = nn_select_move( game );
      valid = othello_game_make_move( game, move );
      if( !valid ) Fail( "move invalid" );
    }

    if( winner==OTHELLO_BIT_BLACK ) black_wins += 1;
    if( winner==OTHELLO_BIT_WHITE ) white_wins += 1;
    if( winner==OTHELLO_GAME_TIED ) draws += 1;
  }

  printf( "Black won: %zu\nWhite won: %zu.\nDraws: %zu\n",
          black_wins, white_wins, draws );

  return 0;
}
