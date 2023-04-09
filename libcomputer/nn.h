#pragma once

#include <stdint.h>

typedef struct othello_game othello_game_t; /* forward decl */

/* A NN player for the othello game */

uint64_t
nn_select_move( othello_game_t const * game );
