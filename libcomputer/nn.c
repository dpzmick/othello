#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../libcommon/common.h"
#include "../libothello/othello.h"

#include "weights.c"

static inline size_t row_major_idx( size_t r, size_t c, size_t nc ) { return nc*r + c; }

/*
   y = x W^T + b

   y: vector of dim (n_outputs,)
   x: vector of dim (n_inputs,)

   W: matrix of dim (n_outputs, n_inputs)
   b: vector of dim (n_outputs,)
*/

static inline void
linear_layer( float const * restrict input,
              size_t                 n_inputs,
              float * restrict       output,
              size_t                 n_outputs,
              float const * restrict weights,
              float const * restrict biases )
{
  for( size_t out_idx = 0; out_idx < n_outputs; ++out_idx ) {
    float acc = 0;
    for( size_t in_idx = 0; in_idx < n_inputs; ++in_idx ) {
      size_t idx = row_major_idx(out_idx, in_idx, n_inputs);
      acc += input[in_idx] * weights[idx];
    }
    output[out_idx] = acc + biases[out_idx];
  }
}

static inline void
relu( float * x,
      size_t  n )
{
  for( size_t i = 0; i < n; ++i ) {
    // unpredictable so written branch free
    float which[] = { 0.0f, x[i] };
    x[i] = which[x[i] > 0.0f];
  }
}

static inline void
sigmoid( float * x,
         size_t  n )
{
  for( size_t i = 0; i < n; ++i ) {
    x[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

float
pred_board_quality( uint64_t white,
                    uint64_t black )
{
  float input[128];
  float hidden1[512];
  float hidden2[512];
  float out[1];

  size_t idx = 0;
  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      bool occupied = white & othello_bit_mask( x, y );
      input[idx++] = occupied ? 1.0f : 0.0f;
    }
  }

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      bool occupied = black & othello_bit_mask( x, y );
      input[idx++] = occupied ? 1.0f : 0.0f;
    }
  }

  linear_layer( input, 128, hidden1, 512, l1_weights, l1_biases );
  relu( hidden1, 512 );
  linear_layer( hidden1, 512, hidden2, 512, l2_weights, l2_biases );
  relu( hidden1, 512 );
  linear_layer( hidden2, 512, out, 1, l3_weights, l3_biases );
  sigmoid( out, 1 );

  return out[0];
}

uint64_t
nn_select_move( othello_game_t const *     game,
                othello_move_ctx_t const * ctx )
{
  if( game->curr_player!=OTHELLO_BIT_WHITE ) Fail( "Computer only plays as white" );

  uint64_t moves      = ctx->own_moves;
  uint64_t n_moves    = ctx->n_own_moves;
  if( n_moves==0 ) return OTHELLO_MOVE_PASS;

  uint64_t best_move  = OTHELLO_MOVE_PASS;
  float    best_score = 0.0f;

  for( uint64_t move_idx = 0; move_idx < n_moves; ++move_idx ) {
    othello_game_t updated_game[1] = { *game };
    uint64_t       move            = keep_ith_set_bit( moves, move_idx );
    bool           valid           = othello_game_make_move( updated_game, ctx, move );
    if( UNLIKELY( !valid ) ) Fail( "attempted to apply invalid move" );

    float score = pred_board_quality( game->white, game->black );
    if( score > best_score ) {
      best_move  = move;
      best_score = score;
    }
  }

  return best_move;
}

void
nn_format_input( othello_game_t const *     game,
                 othello_move_ctx_t const * ctx,
                 float *                    ret )
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
