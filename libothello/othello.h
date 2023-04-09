#pragma once

#include <stdbool.h>
#include <stdint.h>

/* index the bitboard going left to right, so 0,0 prints in upper left corner
   index is 0 based */

static inline uint64_t othello_bit_idx(   uint64_t x, uint64_t y ) { return UINT64_C(63) - (y * UINT64_C(8) + x); }
static inline uint64_t othello_bit_mask(  uint64_t x, uint64_t y ) { return UINT64_C(1) << (othello_bit_idx((x),(y)));    }

enum {
  OTHELLO_BIT_WHITE         = 0,
  OTHELLO_BIT_BLACK         = 1,
  OTHELLO_GAME_TIED         = 2,
  OTHELLO_MOVE_PASS         = 666,
  OTHELLO_GAME_SET_SENTINEL = 0b10101010, /* Stash in curr_player to mark the game as invalid, for use in the hash set */
};

/* Holds both the board state _and_ the game state. Game state is just the current player.

   The first implementation kept board and game state nicely separated, but the
   separation was not assiting in code clarity in the driver code. Jammed them
   back together as stateful game interface works well for the training-set
   generation code even though I don't love the coupling.

   It's also a bit more awkward to test this way since board cannot be tested
   independent of move making. */

typedef struct othello_game {
  uint8_t  curr_player;
  uint64_t white;
  uint64_t black;
} othello_game_t;

/* Initialize the board in a valid state for real othello */

void
othello_game_init( othello_game_t * game );

/* For testing. Doesn't require a board which is reachable from the "real" init
   state as input. */

void
othello_game_init_from_str( othello_game_t * game,
                            uint8_t          next_player,
                            char const *     board_str );

/* Compare the boards of two games _and_ the current player.

   It is possible to reach the same board state with different current player if
   players pass, so this is included in the comparison. */

bool
othello_game_eq( othello_game_t const * a,
                 othello_game_t const * b );

/* Compute a hash of the board and game state */

uint64_t
othello_game_hash( othello_game_t const * game );

void
othello_board_print( othello_game_t const * game );

/*  Check if game is over, return winner if so (else out_winner is untouched) */

bool
othello_game_is_over( othello_game_t const * game,
                      uint8_t *              out_winner );

/* Return all valid moves for current player */

uint64_t
othello_game_all_valid_moves( othello_game_t const * game );

/* Current player makes a move. Move specified by bitboard with single bit set
   (the location of the move) */

bool
othello_game_make_move( othello_game_t * game,
                        uint64_t         move );

/* Play the game till completion, returns winner. Modifies input game. */

uint8_t
othello_game_random_playout( othello_game_t * game,
                             uint64_t         seed );
