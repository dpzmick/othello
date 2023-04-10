#include <stdbool.h>
#include <stdint.h>

#undef __ARM_NEON__
#ifdef __ARM_NEON__
#include <arm_neon.h>
#else
typedef uint64_t uint64x2_t __attribute__((vector_size(128)));
typedef int64_t  int64x2_t  __attribute__((vector_size(128)));

static inline uint64x2_t vandq_u64( uint64x2_t a, uint64x2_t b ) { return a & b; }
static inline uint64x2_t vorrq_u64( uint64x2_t a, uint64x2_t b ) { return a | b; }
static inline uint64x2_t vshlq_u64( uint64x2_t a,  int64x2_t b ) { return a << b; }
#endif

// FIXME must be a better way to do this?
static inline bool
any_non_zero( uint64x2_t v )
{
  return v[0]!=0 || v[1]!=0;
}

static inline uint64_t
_gen_shift_moves_partial( uint64x2_t owns,
                          uint64x2_t opps,
                          uint64x2_t empties,
                          int64x2_t  shifts,
                          uint64x2_t masks )
{
  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it

     own, and opp are broadcast */

  uint64x2_t candidates = vandq_u64(
    masks,
    vandq_u64(
      opps,
      vshlq_u64( owns, shifts ) ) );

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  uint64x2_t lane_moves = { 0, 0 };
  while( any_non_zero( candidates ) ) {
    /* add to moves any empty cells <direction> of a current candidate */

    lane_moves = vorrq_u64(
      lane_moves,
      vandq_u64(
        empties,
        vshlq_u64( candidates, shifts ) ) );

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = vandq_u64(
      masks,
      vandq_u64(
        opps,
        vshlq_u64( candidates, shifts ) ) );

  }

  return lane_moves[0] | lane_moves[1];
}

uint64_t
_all_valid_moves( othello_game_t const * game,
                  uint8_t                player )
{
  uint64_t own   = player==OTHELLO_BIT_WHITE ? game->white : game->black;
  uint64_t opp   = player==OTHELLO_BIT_WHITE ? game->black : game->white;
  uint64_t empty = (~own & ~opp);

  uint64x2_t owns    = { own, own };
  uint64x2_t opps    = { opp, opp };
  uint64x2_t empties = { empty, empty };

  int64x2_t shifts_a = {
    INT64_C(1), /* left */
    INT64_C(8), /* up */
  };

  int64x2_t shifts_b = {
    INT64_C(9), /* up-left */
    INT64_C(7), /* up-right */
  };

  int64x2_t shifts_c = {
    INT64_C(-1), /* right */
    INT64_C(-8), /* down */
  };

  int64x2_t shifts_d = {
    INT64_C(-9), /* down-left */
    INT64_C(-7), /* down-left */
  };

  uint64x2_t masks_a = {
    UINT64_C(0x7e7e7e7e7e7e7e7e), /* left */
    UINT64_C(0xffffffffffff00),   /* up */
  };

  uint64x2_t masks_b = {
    UINT64_C(0x7e7e7e7e7e7e00),   /* up-left */
    UINT64_C(0x7e7e7e7e7e7e00),   /* up-right */
  };

  uint64x2_t masks_c = {
    UINT64_C(0x7e7e7e7e7e7e7e7e), /* right */
    UINT64_C(0xffffffffffff00),   /* down */
  };

  uint64x2_t masks_d = {
    UINT64_C(0x7e7e7e7e7e7e00),   /* down-left */
    UINT64_C(0x7e7e7e7e7e7e00),   /* down-right */
  };

  uint64_t moves = 0;
  moves |= _gen_shift_moves_partial( owns, opps, empties, shifts_a, masks_a );
  moves |= _gen_shift_moves_partial( owns, opps, empties, shifts_b, masks_b );
  moves |= _gen_shift_moves_partial( owns, opps, empties, shifts_c, masks_c );
  moves |= _gen_shift_moves_partial( owns, opps, empties, shifts_d, masks_d );

  return moves;
}
