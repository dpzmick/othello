#pragma once

#include "common.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// index going left to right, so 0,0 prints in upper left corner
// IDX is 0 based
#define BIT_IDX(x,y)  (UINT64_C(63) - (y * UINT64_C(8) + x))
#define BIT_MASK(x,y) (UINT64_C(1) << (BIT_IDX((x),(y))))

// upper right goes to highest bit
static_assert(BIT_IDX(0,0)==63, "indexing is not correct");
static_assert(BIT_IDX(7,7)==0,  "indexing is not correct");

typedef enum {
  PLAYER_WHITE = 0,
  PLAYER_BLACK = 1,
  GAME_TIED = 2,
} player_t;

/* Game board structure */
typedef struct {
  uint64_t white;
  uint64_t black;
} board_t;

static inline void
board_init( board_t * board )
{
  board->white = BIT_MASK(3,3) | BIT_MASK(4,4);
  board->black = BIT_MASK(3,4) | BIT_MASK(4,3);
}

static inline bool
board_eq( board_t const * a,
          board_t const * b )
{
  return a->white == b->white && a->black == b->black;
}

static inline uint64_t
board_white_stones( board_t const * board )
{
  return (uint64_t)__builtin_popcountll( board->white );
}

static inline uint64_t
board_black_stones( board_t const * board )
{
  return (uint64_t)__builtin_popcountll( board->black );
}

static inline uint64_t
board_total_stones( board_t const * board )
{
  return board_white_stones( board ) + board_black_stones( board );
}

static inline void
board_init_random( board_t * board,
                   uint64_t  seed )
{
  // very slow, for testing only
  // could probably do much better

  board->white = 0;
  board->black = 0;
  for( size_t x = 0; x < 8; ++x ) {
    for( size_t y = 0; y < 8; ++y ) {
      uint64_t random = hash_u64( seed+x+y ) % 3; // hehe
      if( random==0 ) continue;
      if( random==1 ) board->white |= BIT_MASK( x,y );
      if( random==2 ) board->black |= BIT_MASK( x,y );
    }
  }
}

static inline void
board_init_random_n_set( board_t * board,
                         uint64_t  seed,
                         size_t    n )
{
  // very slow, for testing only
  // could probably do much better
  // obviously biased towards upper corner
  // and obviously extrememly slow

  board->white = 0;
  board->black = 0;
  size_t cnt = 1;

  assert(n<=64);

  for( size_t j = 0; j < n; ++j ) {
    // pick where to put the piece
    for( size_t try = 1; 1; ++try ) {
      uint64_t x    = hash_u64( seed*cnt++ ) % 8;
      uint64_t y    = hash_u64( seed*cnt++ ) % 8;
      uint64_t mask = BIT_MASK( x,y );

      if( board->white&mask || board->black&mask ) continue;

      uint64_t * which = j%2 ? &board->black : &board->white;
      *which |= mask;
      break;
    }
  }
}

/* "Private" functions but exposed for testing */

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

static inline uint64_t
board_get_all_moves( board_t const * board,
                     player_t        player )
{
  uint64_t own = player==PLAYER_WHITE ? board->white : board->black;
  uint64_t opp = player==PLAYER_WHITE ? board->black : board->white;

  uint64_t moves = 0;
  moves |= board_right_moves( own, opp );
  moves |= board_left_moves( own, opp );
  moves |= board_up_moves( own, opp );
  moves |= board_down_moves( own, opp );
  moves |= board_down_right_moves( own, opp );
  moves |= board_down_left_moves( own, opp );
  moves |= board_up_right_moves( own, opp );
  moves |= board_up_left_moves( own, opp );

  return moves;
}

static inline bool
board_make_move( board_t * board,
                 player_t  player,
                 uint64_t  mx,
                 uint64_t  my )
{
  uint64_t move = BIT_MASK( mx, my );
  if( (move & board_get_all_moves( board, player )) == 0 ) return false;

  uint64_t * own_p = player==PLAYER_WHITE ? &board->white : &board->black;
  uint64_t * opp_p = player==PLAYER_WHITE ? &board->black : &board->white;

  uint64_t own = *own_p;
  uint64_t opp = *opp_p;
  uint64_t empty = (~own & ~opp);

  // n,s,e,w,ne,nw,se,sw
  int64_t x_adjs[8] = {0,0,1,-1,1,-1,1,-1};
  int64_t y_adjs[8] = {1,-1,0,0,1,1,-1,-1};
  for( size_t d = 0; d < 8; ++d ) {
    int64_t dx = x_adjs[d];
    int64_t dy = y_adjs[d];

    int64_t signed_x = (int64_t)mx+dx;
    int64_t signed_y = (int64_t)my+dy;

    // scan in this direction until we hit:
    // 1. empty
    // 2. our own piece
    //
    // Flip pieces we find along the way.

    uint64_t flips = 0;
    bool hit_own = false;
    while( 1 ) {
      if( signed_x < 0 || signed_x >= 8 ) break;
      if( signed_y < 0 || signed_y >= 8 ) break;

      uint64_t x = (uint64_t)signed_x;
      uint64_t y = (uint64_t)signed_y;

      if( own & BIT_MASK( x, y ) ) {
        hit_own = true;
        break;
      }

      if( empty & BIT_MASK( x, y ) ) {
        break;
      }

      flips |= BIT_MASK( x, y );

      signed_x += dx;
      signed_y += dy;
    }

    // do the flips
    if( hit_own ) {
      opp &= ~flips;
      own |= flips;
    }
  }

  // FIXME optimize the above to take advantage of all the clever work we've
  // done to find valid moves?

  *opp_p = opp;
  *own_p = own | BIT_MASK( mx, my ); // include new move

  return true;
}

// FIXME bind to lua
static inline bool
board_is_game_over( board_t const * board,
                    player_t *      out_winner )
{
  // check that neither player can move
  uint64_t white_moves = board_get_all_moves( board, PLAYER_WHITE );
  uint64_t black_moves = board_get_all_moves( board, PLAYER_BLACK );

  if( white_moves==0 && black_moves==0 ) {
    uint64_t white_cnt = board_white_stones( board );
    uint64_t black_cnt = board_black_stones( board );
    if( white_cnt==black_cnt ) {
      *out_winner = GAME_TIED;
    } else if( white_cnt>black_cnt ) {
      *out_winner = PLAYER_WHITE;
    } else {
      *out_winner = PLAYER_BLACK;
    }
    return true;
  }
  else {
    return false;
  }
}

static inline void
board_print( board_t const * board )
{
  printf("  | 0 ");
  for( size_t x = 1; x < 8; ++x ) {
    printf("%zu ", x);
  }
  printf("\n--+----------------\n");

  for( size_t y = 0; y < 8; ++y ) {
    printf("%zu | ", y );

    for( size_t x = 0; x < 8; ++x ) {
      uint8_t bit_white = (board->white & BIT_MASK(x,y)) != 0;
      uint8_t bit_black = (board->black & BIT_MASK(x,y)) != 0;

      if( bit_white && bit_black ) {
        printf( "X " ); // invalid
      }
      else if( bit_white ) {
        printf( "W " );
      }
      else if( bit_black ) {
        printf( "B " );
      }
      else {
        printf( "  " );
      }
    }
    printf("\n");
  }
}

static inline player_t
play_randomly( board_t  board,
               player_t next_player,
               uint64_t seed )
{
  size_t cnt = 0;

  for( ;; next_player = !next_player, cnt += 1 ) {
    player_t winner;
    if( board_is_game_over( &board, &winner ) ) return winner;

    uint64_t moves = board_get_all_moves( &board, next_player );
    if( !moves ) continue;

    // pick a move at random. FIXME use better random number generator here
    // FIXME optimize the selector
    uint64_t n_moves       = (uint64_t)__builtin_popcountll( moves );
    uint64_t rand_move_idx = hash_u64( seed+cnt ) % n_moves; // FIXME modulo bad

    // extract that move
    for( uint64_t x = 0; x < 8; ++x ) {
      for( uint64_t y = 0; y < 8; ++y ) {
        if( 0==(moves&BIT_MASK(x,y)) ) continue;
        if( rand_move_idx--==0 ) {
          // make the move
          bool ret = board_make_move( &board, next_player, x, y );
          assert( ret );
        }
      }
    }
  }
}
