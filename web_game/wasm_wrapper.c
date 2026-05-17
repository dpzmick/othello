#include "../libcommon/common.h"
#include "../libcomputer/mcts.h"
#include "../libcomputer/nn_policy.h"
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

/* Human is black (black moves first in Othello), computer is white. After
   the human plays (or auto-passes), advance the game through any forced
   passes and computer responses until either the game is over or it's the
   human's turn with at least one legal move. */
EMSCRIPTEN_KEEPALIVE
void
othello_wrap_play_at( othello_wrap_t * wrap,
                      int              x,
                      int              y )
{
  uint8_t winner;
  othello_move_ctx_t ctx[1];

  if( !othello_game_start_move( wrap->game, ctx, &winner ) ) return; /* game over */

  /* Human's turn: either auto-pass (no legal moves) or validate the click. */
  if( wrap->game->curr_player == OTHELLO_BIT_BLACK ) {
    if( ctx->n_own_moves == 0 ) {
      othello_game_make_move( wrap->game, ctx, OTHELLO_MOVE_PASS );
    } else {
      uint64_t move = othello_bit_mask( (uint64_t)x, (uint64_t)y );
      if( (move & ctx->own_moves) == 0 ) return; /* ignore invalid click */
      othello_game_make_move( wrap->game, ctx, move );
    }
  }

  /* Advance computer turns + any forced human passes until human has a
     real move to make. Mode 3 (no AI) returns immediately so the UI can
     hand control back to the other human. */
  if( wrap->mode == 3 ) return;

  while( 1 ) {
    if( !othello_game_start_move( wrap->game, ctx, &winner ) ) return;

    if( wrap->game->curr_player == OTHELLO_BIT_BLACK ) {
      if( ctx->n_own_moves > 0 ) return;       /* hand back to UI */
      othello_game_make_move( wrap->game, ctx, OTHELLO_MOVE_PASS );
      continue;
    }

    /* Computer (white). */
    if( ctx->n_own_moves == 0 ) {
      othello_game_make_move( wrap->game, ctx, OTHELLO_MOVE_PASS );
      continue;
    }

    uint64_t computer_move;
    if( wrap->mode == 0 )      computer_move = mcts_select_move( wrap->mcts, wrap->game, ctx );
    else if( wrap->mode == 1 ) computer_move = nn_policy_select_move( wrap->game, ctx );
    else                       Fail( "not implemented" );
    othello_game_make_move( wrap->game, ctx, computer_move );
  }
}

/* Expose game state to the UI so it can render "Pass" affordances and
   end-of-game messages without having to inspect each cell. */
EMSCRIPTEN_KEEPALIVE
int
othello_wrap_n_valid_moves( othello_wrap_t const * wrap )
{
  uint8_t winner;
  othello_move_ctx_t ctx[1];
  if( !othello_game_start_move( wrap->game, ctx, &winner ) ) return 0;
  return (int)ctx->n_own_moves;
}

EMSCRIPTEN_KEEPALIVE
int
othello_wrap_game_over( othello_wrap_t const * wrap )
{
  uint8_t winner;
  othello_move_ctx_t ctx[1];
  return !othello_game_start_move( wrap->game, ctx, &winner );
}
