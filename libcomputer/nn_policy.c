#include "nn_policy.h"
#include "nn.h"  /* for nn_format_input */

#include "../libcommon/common.h"
#include "../libothello/othello.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Architecture constants and weight arrays. */
#include "weights_policy.c"

static inline size_t
row_major_idx( size_t r, size_t c, size_t nc )
{
  return nc * r + c;
}

static inline void
linear_layer( float const * restrict input,
              size_t                 n_inputs,
              float * restrict       output,
              size_t                 n_outputs,
              float const * restrict weights,
              float const * restrict biases )
{
  for( size_t out_idx = 0; out_idx < n_outputs; ++out_idx ) {
    float acc = biases[out_idx];
    for( size_t in_idx = 0; in_idx < n_inputs; ++in_idx ) {
      acc += input[in_idx] * weights[row_major_idx( out_idx, in_idx, n_inputs )];
    }
    output[out_idx] = acc;
  }
}

static inline void
relu( float * x, size_t n )
{
  for( size_t i = 0; i < n; ++i ) {
    if( x[i] < 0.0f ) x[i] = 0.0f;
  }
}

uint64_t
nn_policy_select_move( othello_game_t const *     game,
                       othello_move_ctx_t const * ctx )
{
  uint64_t valid_moves = ctx->own_moves;
  uint64_t n_moves     = ctx->n_own_moves;

  if( n_moves == 0 ) return OTHELLO_MOVE_PASS;

  /* Run the forward pass. Buffers sized at compile time from weights_policy.c
     constants. */
  float input[NN_POLICY_INPUT_SHAPE];
  float hidden1[NN_POLICY_N1];
  float hidden2[NN_POLICY_N2];
  float logits[NN_POLICY_N_OUTPUTS];

  /* nn_format_input lays out: player(1) + valid_moves(64) + white(64) + black(64)
     -- with no lookback that's 193 floats, matching INPUT_SHAPE. */
  nn_format_input( game, ctx, NULL, 0, input );

  linear_layer( input,   NN_POLICY_INPUT_SHAPE, hidden1, NN_POLICY_N1,
                nn_policy_l1_weights, nn_policy_l1_biases );
  relu( hidden1, NN_POLICY_N1 );

  linear_layer( hidden1, NN_POLICY_N1, hidden2, NN_POLICY_N2,
                nn_policy_l2_weights, nn_policy_l2_biases );
  relu( hidden2, NN_POLICY_N2 );

  linear_layer( hidden2, NN_POLICY_N2, logits, NN_POLICY_N_OUTPUTS,
                nn_policy_l3_weights, nn_policy_l3_biases );

  /* Argmax over valid moves only. Policy index i = move_x + move_y*8, matching
     the one-hot layout in data_gen_policy.c. */
  size_t best_idx  = 64; /* sentinel */
  float  best_val  = -INFINITY;
  for( size_t i = 0; i < 64; ++i ) {
    uint64_t x    = i % 8;
    uint64_t y    = i / 8;
    uint64_t mask = othello_bit_mask( x, y );
    if( !(valid_moves & mask) ) continue;
    if( logits[i] > best_val ) {
      best_val = logits[i];
      best_idx = i;
    }
  }

  if( UNLIKELY( best_idx == 64 ) ) Fail( "no valid move found despite n_moves > 0" );

  uint64_t x = best_idx % 8;
  uint64_t y = best_idx / 8;
  return othello_bit_mask( x, y );
}
