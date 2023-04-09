#include "../libothello/othello.h"
#include "../libcomputer/mcts.h"
#include "../libcomputer/nn.h"

#include <emscripten.h>
#include <stdlib.h>

#define MCTS_TRIALS 1000
#define MCTS_NODES  8192

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

  mcts_state_init( ret->mcts, MCTS_TRIALS, MCTS_NODES );
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
othello_wrap_board_at( othello_wrap_t const * wrap,
                       int                    x,
                       int                    y )
{
  uint64_t white       = wrap->game->white;
  uint64_t black       = wrap->game->black;
  uint64_t valid_moves = othello_game_all_valid_moves( wrap->game );
  uint64_t mask        = othello_bit_mask( (uint64_t)x, (uint64_t)y );


  // FIXME flag valid moves here too?

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
  uint64_t move        = othello_bit_mask( (uint64_t)x, (uint64_t)y );
  uint64_t valid_moves = othello_game_all_valid_moves( wrap->game );
  if( (move&valid_moves) == 0 ) return; // just don't do anything

  othello_game_make_move( wrap->game, move );

  if( wrap->mode==0 ) {
    uint64_t computer_move = mcts_select_move( wrap->mcts, wrap->game );
    othello_game_make_move( wrap->game, computer_move );
  }
  else if( wrap->mode == 1 ) {
    uint64_t computer_move = nn_select_move( wrap->game );
    othello_game_make_move( wrap->game, computer_move );
  }
}
