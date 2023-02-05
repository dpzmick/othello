#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// index going left to right, so 0,0 prints in upper left corner
// IDX is 0 based
#define BIT_IDX(x,y)  (UINT64_C(63) - (y * UINT64_C(8) + x))
#define BIT_MASK(x,y) (UINT64_C(1) << (BIT_IDX((x),(y))))

// upper right goes to highest bit
static_assert(BIT_IDX(0,0)==63, "indexing is not correct");
static_assert(BIT_IDX(7,7)==0,  "indexing is not correct");

/* useful for debugging and inspecting boards */

static inline uint64_t
bitboard_from_rows( uint8_t rows[8] )
{
  uint64_t ret = 0;
  for( size_t row = 0; row < 8; ++row ) {
    uint64_t shift = 64 - 8*(row+1);
    ret |= ((uint64_t)rows[row]) << shift;
  }
  return ret;
}

/* Game board structure */
typedef struct {
  uint64_t white;
  uint64_t black;
} board_t;

static inline void
board_zero( board_t * board )
{
  board->white = 0;
  board->black = 0;
}

static inline void
board_init( board_t * board )
{
  board->white = BIT_MASK(3,3) | BIT_MASK(4,4);
  board->black = BIT_MASK(3,4) | BIT_MASK(4,3);
}

static inline board_t
new_board_from_str( char const * str )
{
  uint64_t white = 0;
  uint64_t black = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      assert(*str == 'B' || *str == 'W' || *str == '.');
      switch( *str ) {
        case 'W': white |= BIT_MASK(x,y); break;
        case 'B': black |= BIT_MASK(x,y); break;
        case '.': break;
      }
      str++;
    }
    /* assert(*str == '\n'); */
    /* str++; // skip newline */
  }

  return (board_t){ .white=white, .black=black };
}

void
board_print( board_t const * board );

static inline uint64_t
board_gen_moves_right_shift( uint64_t own,
                             uint64_t opp,
                             uint64_t direction_shift,
                             uint64_t edge_mask )
{
  /* discover and save the set of empty cells */
  uint64_t empty = (~own & ~opp);

  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it */

  uint64_t candidates = edge_mask & (opp & (own >> direction_shift));

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  uint64_t moves = 0;
  while( candidates ) {
    /* add to moves any empty cells <direction> of a current candidate */
    moves |= empty & (candidates >> direction_shift);

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = edge_mask & (opp & (candidates >> direction_shift));
  }

  return moves;
}

static inline uint64_t
board_gen_moves_left_shift( uint64_t own,
                            uint64_t opp,
                            uint64_t direction_shift,
                            uint64_t edge_mask )
{
  /* discover and save the set of empty cells */
  uint64_t empty = (~own & ~opp);

  /* start with every space which:
     1. contains an opponent stone
     2. has one of our stones to the <direction> of it */

  uint64_t candidates = edge_mask & (opp & (own << direction_shift));

  /* slide each canidate in <direction> until:
     1. one of our candidate bits is intersecting an empty cell (outflank the line)
     2. or we find our own cell (invalid move) */

  uint64_t moves = 0;
  while( candidates ) {
    /* add to moves any empty cells <direction> of a current candidate */
    moves |= empty & (candidates << direction_shift);

    /* update candiates to include any cells left of a current candidate which
       are occupied by enemy (propgate left). We cannot go off edge of board, so
       apply the mask */

    candidates = edge_mask & (opp & (candidates << direction_shift));
  }

  return moves;
}

static inline uint64_t
board_right_moves( uint64_t own,
                   uint64_t opp )
{
  uint64_t left_and_right_edges = UINT64_C(0x7e7e7e7e7e7e7e7e);
  return board_gen_moves_right_shift( own, opp, 1, left_and_right_edges );
}

static inline uint64_t
board_left_moves( uint64_t own,
                  uint64_t opp )
{
  uint64_t left_and_right_edges = UINT64_C(0x7e7e7e7e7e7e7e7e);
  return board_gen_moves_left_shift( own, opp, 1, left_and_right_edges );
}

static inline uint64_t
board_up_moves( uint64_t own,
                uint64_t opp )
{
  uint64_t top_and_bottom = UINT64_C(0xffffffffffff00);
  return board_gen_moves_left_shift( own, opp, 8, top_and_bottom );
}

static inline uint64_t
board_down_moves( uint64_t own,
                  uint64_t opp )
{
  uint64_t top_and_bottom = UINT64_C(0xffffffffffff00);
  return board_gen_moves_right_shift( own, opp, 8, top_and_bottom );
}

static inline uint64_t
board_down_left_moves( uint64_t own,
                       uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return board_gen_moves_right_shift( own, opp, 9, mask );
}

static inline uint64_t
board_down_right_moves( uint64_t own,
                       uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return board_gen_moves_right_shift( own, opp, 7, mask );
}

static inline uint64_t
board_up_left_moves( uint64_t own,
                     uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return board_gen_moves_left_shift( own, opp, 9, mask );
}


static inline uint64_t
board_up_right_moves( uint64_t own,
                      uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return board_gen_moves_left_shift( own, opp, 7, mask );
}
