#include <stdint.h>

typedef uint64_t v4u64_t __attribute__ ((vector_size (256)));

static inline uint64_t
_gen_left_shift_moves( uint64_t own,
                       uint64_t opp,
                       uint64_t empty )
{
  v4u64_t shifts = {
    UINT64_C(1), /* left */
    UINT64_C(8), /* up */
    UINT64_C(9), /* up-left */
    UINT64_C(7), /* up-right */
  };

  v4u64_t masks = {
    UINT64_C(0x7e7e7e7e7e7e7e7e), /* left */
    UINT64_C(0xffffffffffff00),   /* up */
    UINT64_C(0x7e7e7e7e7e7e00),   /* up-left */
    UINT64_C(0x7e7e7e7e7e7e00),   /* up-right */
  };

  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it

     own, and opp are broadcast */

  v4u64_t candidates = masks & (opp & (own << shifts));

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  v4u64_t lane_moves = { 0, 0, 0, 0 };
  while( candidates[0]!=0 || candidates[1]!=0 || candidates[2]!=0 || candidates[3]!=0 ) {
    /* add to moves any empty cells <direction> of a current candidate */
    lane_moves |= empty & (candidates << shifts);

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = masks & (opp & (candidates << shifts));
  }

  return lane_moves[0] | lane_moves[1] | lane_moves[2] | lane_moves[3];
}

static inline uint64_t
_gen_right_shift_moves( uint64_t own,
                        uint64_t opp,
                        uint64_t empty )
{
  v4u64_t shifts = {
    UINT64_C(1), /* right */
    UINT64_C(8), /* down */
    UINT64_C(9), /* down-left */
    UINT64_C(7), /* down-left */
  };

  v4u64_t masks = {
    UINT64_C(0x7e7e7e7e7e7e7e7e), /* right */
    UINT64_C(0xffffffffffff00),   /* down */
    UINT64_C(0x7e7e7e7e7e7e00),   /* down-left */
    UINT64_C(0x7e7e7e7e7e7e00),   /* down-right */
  };

  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it

     own, and opp are broadcast */

  v4u64_t candidates = masks & (opp & (own >> shifts));

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  v4u64_t lane_moves = { 0, 0, 0, 0 };
  while( candidates[0]!=0 || candidates[1]!=0 || candidates[2]!=0 || candidates[3]!=0 ) {
    /* add to moves any empty cells <direction> of a current candidate */
    lane_moves |= empty & (candidates >> shifts);

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = masks & (opp & (candidates >> shifts));
  }

  return lane_moves[0] | lane_moves[1] | lane_moves[2] | lane_moves[3];
}

uint64_t
_all_valid_moves( othello_game_t const * game,
                  uint8_t                player )
{
  uint64_t own   = player==OTHELLO_BIT_WHITE ? game->white : game->black;
  uint64_t opp   = player==OTHELLO_BIT_WHITE ? game->black : game->white;
  uint64_t empty = (~own & ~opp);

  uint64_t moves = 0;
  moves |= _gen_right_shift_moves( own, opp, empty );
  moves |= _gen_left_shift_moves( own, opp, empty );

  return moves;
}
