#include "../libcommon/common.h"
#include "../libcomputer/mcts.h"
#include "../libcomputer/nn.h"
#include "../libothello/othello.h"

#include <emscripten.h>
#include <stdlib.h>

#define MCTS_TRIALS 7000
#define MCTS_NODES  1<<15

typedef struct {
  othello_game_t game[1];
  mcts_state_t * mcts;
  int            mode;
} othello_wrap_t;

EMSCRIPTEN_KEEPALIVE
othello_wrap_t *
new_othello_wrap( int mode )
{
  othello_wrap_t * ret = malloc( sizeof(othello_wrap_t) );
  if( !ret ) return NULL;

  ret->mcts = malloc( mcts_state_size( MCTS_NODES ) );
  if( !ret->mcts ) {
    free( ret );
    return NULL;
  }

  mcts_state_init( ret->mcts, MCTS_TRIALS, OTHELLO_BIT_WHITE, wallclock(), MCTS_NODES );
  othello_game_init( ret->game );

  ret->mode = mode;

  return ret;
}

EMSCRIPTEN_KEEPALIVE
void
delete_othello_wrap( othello_wrap_t * wrap )
{
  free( wrap );
}

EMSCRIPTEN_KEEPALIVE
int
othello_wrap_turn( othello_wrap_t * wrap )
{
  return wrap->game->curr_player;
}

EMSCRIPTEN_KEEPALIVE
int
othello_wrap_board_at( othello_wrap_t const * wrap,
                       int                    x,
                       int                    y )
{
  uint64_t white       = wrap->game->white;
  uint64_t black       = wrap->game->black;
  uint64_t valid_moves = othello_game_all_valid_moves( wrap->game );
  uint64_t mask        = othello_bit_mask( (uint64_t)x, (uint64_t)y );

  if( mask&white )       return 1;
  if( mask&black )       return -1;
  if( valid_moves&mask ) return 2;
  else                   return 0;
}

EMSCRIPTEN_KEEPALIVE
void
othello_wrap_play_at( othello_wrap_t * wrap,
                      int              x,
                      int              y )
{
  uint64_t move = othello_bit_mask( (uint64_t)x, (uint64_t)y );

  uint8_t winner;
  othello_move_ctx_t ctx[1];
  if( !othello_game_start_move( wrap->game, ctx, &winner ) ) {
    return; // just don't do anything
  }

  uint64_t valid_moves = ctx->own_moves;
  if( (move&valid_moves) == 0 ) {
    return; // just don't do anything
  }

  bool valid = othello_game_make_move( wrap->game, ctx, move );
  if( !valid ) Fail( "should not have failed once we got here" );

  /* start the computer move */
  if( !othello_game_start_move( wrap->game, ctx, &winner ) ) {
    return; // no moves left to make
  }

  if( wrap->mode==0 ) {
    uint64_t computer_move = mcts_select_move( wrap->mcts, wrap->game, ctx );
    othello_game_make_move( wrap->game, ctx, computer_move );
  }
  else if( wrap->mode == 1 ) {
    uint64_t computer_move = nn_select_move( wrap->game, ctx );
    othello_game_make_move( wrap->game, ctx, computer_move );
  }
  else if( wrap->mode == 2 ) {
    Fail( "not implemented" );
  }
  else if( wrap->mode == 3 ) {
    // no computer
  }
}
