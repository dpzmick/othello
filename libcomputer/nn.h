#pragma once

#include <stdint.h>

typedef struct othello_game     othello_game_t; /* forward decl */
typedef struct othello_move_ctx othello_move_ctx_t; /* forward decl */

/* A NN player for the othello game */

uint64_t
nn_select_move( othello_game_t const *     game,
                othello_move_ctx_t const * ctx );

float
pred_board_quality( uint64_t white,
                    uint64_t black );


/* Format the NN input into the ret buffer. Buffer must be the right size.
   ctx is required because we're including the set of valid moves */
void
nn_format_input( othello_game_t const *     game,
                 othello_move_ctx_t const * ctx,
                 float *                    ret );
