#include <stdbool.h>
#include <stdint.h>

static inline uint64x4_t
_gen_left_shift_moves( uint64x4_t owns,
                       uint64x4_t opps,
                       uint64x4_t empties,
                       uint64x4_t shifts,
                       uint64x4_t masks )
{
  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it */

  uint64x4_t candidates = masks & (opps & (owns << shifts));

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  uint64x4_t lane_moves = { 0, 0, 0, 0 };

  /* Instead of searching until we run out of candidates (because checking both
     lanes and reducing isn't vector friendly), just search the max distance we
     could possibly go,*/

  for( size_t i = 0; i < 8; ++i ) {
    /* add to moves any empty cells <direction> of a current candidate */

    lane_moves |= empties & (candidates << shifts);

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = masks & (opps & (candidates << shifts));
  }

  return lane_moves;
}

static inline uint64x4_t
_gen_right_shift_moves( uint64x4_t owns,
                        uint64x4_t opps,
                        uint64x4_t empties,
                        uint64x4_t shifts,
                        uint64x4_t masks )
{
  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it */

  uint64x4_t candidates = masks & (opps & (owns >> shifts));

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  uint64x4_t lane_moves = { 0, 0, 0, 0 };

  /* Instead of searching until we run out of candidates (because checking both
     lanes and reducing isn't vector friendly), just search the max distance we
     could possibly go,*/

  for( size_t i = 0; i < 8; ++i ) {
    /* add to moves any empty cells <direction> of a current candidate */

    lane_moves |= empties & (candidates >> shifts);

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = masks & (opps & (candidates >> shifts));
  }

  return lane_moves;
}

static inline uint64_t
_all_valid_moves( othello_game_t const * game,
                  uint8_t                player )
{
  uint64_t own   = player==OTHELLO_BIT_WHITE ? game->white : game->black;
  uint64_t opp   = player==OTHELLO_BIT_WHITE ? game->black : game->white;
  uint64_t empty = (~own & ~opp);

  uint64x4_t owns    = { own, own, own, own };
  uint64x4_t opps    = { opp, opp, opp, opp };
  uint64x4_t empties = { empty, empty, empty, empty };

  uint64x4_t left_shifts = {
    INT64_C(1), /* left */
    INT64_C(8), /* up */
    INT64_C(9), /* up-left */
    INT64_C(7), /* up-right */
  };

  uint64x4_t left_masks = {
    UINT64_C(0x7e7e7e7e7e7e7e7e), /* left */
    UINT64_C(0xffffffffffff00),   /* up */
    UINT64_C(0x7e7e7e7e7e7e00),   /* up-left */
    UINT64_C(0x7e7e7e7e7e7e00),   /* up-right */
  };

  uint64x4_t right_shifts = {
    UINT64_C(1), /* right */
    UINT64_C(8), /* down */
    UINT64_C(9), /* down-left */
    UINT64_C(7), /* down-left */
  };

  uint64x4_t right_masks = {
    UINT64_C(0x7e7e7e7e7e7e7e7e), /* right */
    UINT64_C(0xffffffffffff00),   /* down */
    UINT64_C(0x7e7e7e7e7e7e00),   /* down-left */
    UINT64_C(0x7e7e7e7e7e7e00),   /* down-right */
  };

  uint64x4_t moves = { 0, 0 };
  moves |= _gen_left_shift_moves( owns, opps, empties, left_shifts, left_masks );
  moves |= _gen_right_shift_moves( owns, opps, empties, right_shifts, right_masks );

  return moves[0] | moves[1] | moves[2] | moves[3];
}
