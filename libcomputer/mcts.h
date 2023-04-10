#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct othello_game     othello_game_t; /* forward decl */
typedef struct othello_move_ctx othello_move_ctx_t; /* forward decl */

/* A MCTS player for the othello game */

typedef struct mcts_state mcts_state_t;

size_t
mcts_state_size( size_t n_boards_memory );

void
mcts_state_init( mcts_state_t * state,
                 size_t         trials,
                 uint8_t        play_as,
                 uint64_t       seed,
                 size_t         n_boards );

uint64_t
mcts_select_move( mcts_state_t *             mcts,
                  othello_game_t const *     game,
                  othello_move_ctx_t const * ctx );
