#include "nn_policy.h"

#include "../libcommon/common.h"
#include "../libothello/othello.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Architecture constants and weight arrays. */
#include "weights_policy.c"

/* Default to the two-hidden-layer NN architecture for weights files generated
   before the NN_single support was added. */
#ifndef NN_POLICY_HAS_L2
#define NN_POLICY_HAS_L2 1
#endif

/* For older weights files: L1 consumed the full formatted input. The NN_no_valid
   variant drops the leading valid-moves plane, in which case dump_weights.py
   emits a smaller L1_INPUT_SIZE explicitly. */
#ifndef NN_POLICY_L1_INPUT_SIZE
#define NN_POLICY_L1_INPUT_SIZE NN_POLICY_INPUT_SHAPE
#endif

/* --- input formatting --------------------------------------------------- */

void
nn_format_input( othello_game_t const *     game,
                 othello_move_ctx_t const * ctx,
                 othello_game_t const *     lookback_boards,
                 size_t                     n_lookback_boards,
                 float *                    ret )
{
  /* Canonicalized input layout (no current-player byte):
     1. valid moves (64)
     2. "my" pieces plane (64), where "my" = whichever color is to move
     3. "opp" pieces plane (64)
     [4]. previous boards, same my/opp convention, if any

     The network only ever sees boards from the perspective of the player
     to move; the explicit player byte is gone, and white/black get
     swapped into my/opp based on game->curr_player. */

  uint64_t my_pieces  = (game->curr_player == OTHELLO_BIT_WHITE) ? game->white : game->black;
  uint64_t opp_pieces = (game->curr_player == OTHELLO_BIT_WHITE) ? game->black : game->white;

  size_t idx = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      ret[idx++] = ctx->own_moves & othello_bit_mask( x, y ) ? 1.0f : 0.0f;
    }
  }

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      ret[idx++] = my_pieces & othello_bit_mask( x, y ) ? 1.0f : 0.0f;
    }
  }

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      ret[idx++] = opp_pieces & othello_bit_mask( x, y ) ? 1.0f : 0.0f;
    }
  }

  assert( idx == 64+128 );

  for( size_t i = 0; i < n_lookback_boards; ++i ) {
    othello_game_t const * lb = &lookback_boards[i];
    uint64_t lb_my  = (game->curr_player == OTHELLO_BIT_WHITE) ? lb->white : lb->black;
    uint64_t lb_opp = (game->curr_player == OTHELLO_BIT_WHITE) ? lb->black : lb->white;

    for( size_t y = 0; y < 8; ++y ) {
      for( size_t x = 0; x < 8; ++x ) {
        ret[idx++] = lb_my & othello_bit_mask( x, y ) ? 1.0f : 0.0f;
      }
    }

    for( size_t y = 0; y < 8; ++y ) {
      for( size_t x = 0; x < 8; ++x ) {
        ret[idx++] = lb_opp & othello_bit_mask( x, y ) ? 1.0f : 0.0f;
      }
    }
  }

  assert( idx == 64+128+128*n_lookback_boards );
}

/* --- policy forward pass ----------------------------------------------- */

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
     constants. Explicit zero-init keeps GCC's -Wmaybe-uninitialized happy --
     it can't see through nn_format_input to prove the buffer is filled. */
  float input[NN_POLICY_INPUT_SHAPE]  = {0};
  float hidden1[NN_POLICY_N1]         = {0};
#if NN_POLICY_HAS_L2
  float hidden2[NN_POLICY_N2]         = {0};
#endif
  float logits[NN_POLICY_N_OUTPUTS]   = {0};

  /* nn_format_input lays out: valid_moves(64) + my(64) + opp(64) -- 192
     floats with no lookback, matching NN_POLICY_INPUT_SHAPE. The slice
     offset trims the leading valid plane when the trained net was an
     NN_no_valid (L1_INPUT_SIZE < INPUT_SHAPE). */
  nn_format_input( game, ctx, NULL, 0, input );
  size_t const l1_offset = NN_POLICY_INPUT_SHAPE - NN_POLICY_L1_INPUT_SIZE;

  linear_layer( input + l1_offset, NN_POLICY_L1_INPUT_SIZE, hidden1, NN_POLICY_N1,
                nn_policy_l1_weights, nn_policy_l1_biases );
  relu( hidden1, NN_POLICY_N1 );

#if NN_POLICY_HAS_L2
  linear_layer( hidden1, NN_POLICY_N1, hidden2, NN_POLICY_N2,
                nn_policy_l2_weights, nn_policy_l2_biases );
  relu( hidden2, NN_POLICY_N2 );

  linear_layer( hidden2, NN_POLICY_N2, logits, NN_POLICY_N_OUTPUTS,
                nn_policy_l3_weights, nn_policy_l3_biases );
#else
  linear_layer( hidden1, NN_POLICY_N1, logits, NN_POLICY_N_OUTPUTS,
                nn_policy_l3_weights, nn_policy_l3_biases );
#endif

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
