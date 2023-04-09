#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct othello_game othello_game_t; /* forward decl */

/* A MCTS player for the othello game */

typedef struct mcts_state mcts_state_t;

size_t
mcts_state_size( size_t n_boards_memory );

mcts_state_t *
mcts_state_init( void *   mem,
                 size_t   trials,
                 uint8_t  play_as,
                 uint64_t seed,
                 size_t   n_boards );

uint64_t
mcts_select_move( mcts_state_t *         mcts,
                  othello_game_t const * game );
