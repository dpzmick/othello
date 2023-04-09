#include "othello.h"
#include "../libcommon/hash.h"
#include "../libcommon/common.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static inline uint64_t
_board_gen_moves_right_shift( uint64_t own,
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
_board_gen_moves_left_shift( uint64_t own,
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
_board_right_moves( uint64_t own,
                    uint64_t opp )
{
  uint64_t left_and_right_edges = UINT64_C(0x7e7e7e7e7e7e7e7e);
  return _board_gen_moves_right_shift( own, opp, 1, left_and_right_edges );
}

static inline uint64_t
_board_left_moves( uint64_t own,
                   uint64_t opp )
{
  uint64_t left_and_right_edges = UINT64_C(0x7e7e7e7e7e7e7e7e);
  return _board_gen_moves_left_shift( own, opp, 1, left_and_right_edges );
}

static inline uint64_t
_board_up_moves( uint64_t own,
                 uint64_t opp )
{
  uint64_t top_and_bottom = UINT64_C(0xffffffffffff00);
  return _board_gen_moves_left_shift( own, opp, 8, top_and_bottom );
}

static inline uint64_t
_board_down_moves( uint64_t own,
                   uint64_t opp )
{
  uint64_t top_and_bottom = UINT64_C(0xffffffffffff00);
  return _board_gen_moves_right_shift( own, opp, 8, top_and_bottom );
}

static inline uint64_t
_board_down_left_moves( uint64_t own,
                        uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return _board_gen_moves_right_shift( own, opp, 9, mask );
}

static inline uint64_t
_board_down_right_moves( uint64_t own,
                         uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return _board_gen_moves_right_shift( own, opp, 7, mask );
}

static inline uint64_t
_board_up_left_moves( uint64_t own,
                      uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return _board_gen_moves_left_shift( own, opp, 9, mask );
}

static inline uint64_t
_board_up_right_moves( uint64_t own,
                       uint64_t opp )
{
  // top and bottom and left and right
  uint64_t mask = UINT64_C(0x7e7e7e7e7e7e00);
  return _board_gen_moves_left_shift( own, opp, 7, mask );
}

uint64_t
_all_valid_moves( othello_game_t const * game,
                  uint8_t                player )
{
  uint64_t own = player==OTHELLO_BIT_WHITE ? game->white : game->black;
  uint64_t opp = player==OTHELLO_BIT_WHITE ? game->black : game->white;

  // FIXME could each direction be run in a different vector lane?
  //
  // the logic is the same for all these, just with different masks and
  // different shift amounts/operators.

  uint64_t moves = 0;
  moves |= _board_right_moves( own, opp );
  moves |= _board_left_moves( own, opp );
  moves |= _board_up_moves( own, opp );
  moves |= _board_down_moves( own, opp );
  moves |= _board_down_right_moves( own, opp );
  moves |= _board_down_left_moves( own, opp );
  moves |= _board_up_right_moves( own, opp );
  moves |= _board_up_left_moves( own, opp );

  return moves;
}

void
othello_game_init( othello_game_t * game )
{
  memset( game, 0, sizeof(*game) ); // make sure padding bits are cleared

  game->curr_player = OTHELLO_BIT_BLACK; // black should probably be 0 FIXME
  game->white = othello_bit_mask(3,3) | othello_bit_mask(4,4);
  game->black = othello_bit_mask(3,4) | othello_bit_mask(4,3);
}

void
othello_game_init_from_str( othello_game_t * game,
                            uint8_t          next_player,
                            char const *     str )
{
  memset( game, 0, sizeof(*game) ); // make sure padding bits are cleared

  game->curr_player = next_player;
  game->white       = 0;
  game->black       = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      assert(*str == 'B' || *str == 'W' || *str == '.');
      switch( *str ) {
        case 'W': game->white |= othello_bit_mask(x,y); break;
        case 'B': game->black |= othello_bit_mask(x,y); break;
        case '.': break;
      }
      str++;
    }
  }
}

bool
othello_game_eq( othello_game_t const * a,
                 othello_game_t const * b )
{
  /* there are padding bits in our struct, but we've always memset them to zero
     in initialization. Technically a lil UB */

  return 0==memcmp( a, b, sizeof(*a) );
}

uint64_t
othello_game_hash( othello_game_t const * game )
{
  /* there are padding bits in our struct, but we've always memset them to zero
     in initialization. Technically a lil UB */

  return fd_hash( 0x1a2b3c4d5e6f8aUL, (void*)game, sizeof(*game) );
}

size_t
othello_game_popcount( othello_game_t const * game )
{
  return (size_t)__builtin_popcountll( game->white )
    + (size_t)__builtin_popcountll( game->black );
}

void
othello_board_print( othello_game_t const * game )
{
  printf("  | 0 ");
  for( size_t x = 1; x < 8; ++x ) {
    printf("%zu ", x);
  }
  printf("\n--+----------------\n");

  for( size_t y = 0; y < 8; ++y ) {
    printf("%zu | ", y );

    for( size_t x = 0; x < 8; ++x ) {
      uint8_t bit_white = (game->white & othello_bit_mask(x,y)) != 0;
      uint8_t bit_black = (game->black & othello_bit_mask(x,y)) != 0;

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

bool
othello_game_is_over( othello_game_t const * game,
                      uint8_t *              out_winner )
{
  uint8_t curr_player = game->curr_player;

  uint64_t my_moves  = _all_valid_moves( game, curr_player );
  uint64_t opp_moves = _all_valid_moves( game, !curr_player );

  uint64_t my_moves_cnt  = (uint64_t)__builtin_popcountll( my_moves );
  uint64_t opp_moves_cnt = (uint64_t)__builtin_popcountll( opp_moves );

  /* game is over */
  if( my_moves_cnt==0 && opp_moves_cnt==0 ) {
    uint64_t white_stones = (uint64_t)__builtin_popcountll( game->white ); // FIXME don't redo
    uint64_t black_stones = (uint64_t)__builtin_popcountll( game->black );

    if( white_stones > black_stones ) *out_winner = OTHELLO_BIT_WHITE;
    if( black_stones > white_stones ) *out_winner = OTHELLO_BIT_BLACK;
    else                         *out_winner = OTHELLO_GAME_TIED;

    return true;
  }
  else {
    return false;
  }
}

uint64_t
othello_game_all_valid_moves( othello_game_t const * game )
{
  return _all_valid_moves( game, game->curr_player );
}

// lmao so bad
static inline void
_extract_move( uint64_t   all_moves,
               uint64_t   idx,
               uint64_t * out_x,
               uint64_t * out_y )
{
  for( uint64_t x = 0; x < 8; ++x ) {
    for( uint64_t y = 0; y < 8; ++y ) {
      if( 0==(all_moves&othello_bit_mask(x,y)) ) continue;
      if( idx--==0 ) {
        *out_x = x;
        *out_y = y;
        return;
      }
    }
  }
}

bool
othello_game_make_move( othello_game_t * game,
                        uint64_t         move )
{
  if( move == OTHELLO_MOVE_PASS ) {
    game->curr_player = !game->curr_player;
    return true;
  }

  if( 1 != __builtin_popcountll( move ) ) return false;

  if( (move & othello_game_all_valid_moves( game )) == 0 ) return false;

  uint8_t    player = game->curr_player;
  uint64_t * own_p  = player==OTHELLO_BIT_WHITE ? &game->white : &game->black;
  uint64_t * opp_p  = player==OTHELLO_BIT_WHITE ? &game->black : &game->white;

  uint64_t own = *own_p;
  uint64_t opp = *opp_p;
  uint64_t empty = (~own & ~opp);

  // FIXME this seems like it could be make significantly more efficient
  //
  // it may be possible to precompute flips in each direction then apply
  // precomputed masks for a given board state?

  uint64_t mx = 0, my = 0;
  _extract_move( move, 0, &mx, &my );

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

      if( own & othello_bit_mask( x, y ) ) {
        hit_own = true;
        break;
      }

      if( empty & othello_bit_mask( x, y ) ) {
        break;
      }

      flips |= othello_bit_mask( x, y );

      signed_x += dx;
      signed_y += dy;
    }

    // do the flips
    if( hit_own ) {
      opp &= ~flips;
      own |= flips;
    }
  }

  *opp_p = opp;
  *own_p = own | move;

  game->curr_player = !game->curr_player;
  return true;
}

uint8_t
othello_game_random_playout( othello_game_t * game,
                             uint64_t         seed )
{
  for( size_t cnt = 0;; cnt += 1 ) {
    uint8_t curr_player = game->curr_player;

    // not caling game over function here to avoid computing moves twice
    // FIXME check if it would have inlined anyway?

    uint64_t my_moves  = _all_valid_moves( game, curr_player );
    uint64_t opp_moves = _all_valid_moves( game, !curr_player ); // FIXME don't ask for this unless my_moves_cnt==0?

    uint64_t my_moves_cnt  = (uint64_t)__builtin_popcountll( my_moves );
    uint64_t opp_moves_cnt = (uint64_t)__builtin_popcountll( opp_moves );

    /* game is over */
    if( my_moves_cnt==0 && opp_moves_cnt==0 ) {
      uint64_t white_stones = (uint64_t)__builtin_popcountll( game->white ); // FIXME don't redo
      uint64_t black_stones = (uint64_t)__builtin_popcountll( game->black );

      if( white_stones > black_stones ) return OTHELLO_BIT_WHITE;
      if( black_stones > white_stones ) return OTHELLO_BIT_BLACK;
      else                              return OTHELLO_GAME_TIED;
    }

    /* have to pass, other player can still play */
    if( my_moves_cnt==0 ) {
      bool valid = othello_game_make_move( game, OTHELLO_MOVE_PASS );
      if( !valid ) Fail( "tried to make invalid move" );

      continue;
    }

    // pick a move at random.
    // FIXME modulo isn't great
    uint64_t rand_move_idx = hash_u64( seed+cnt ) % my_moves_cnt;
    uint64_t move          = keep_ith_set_bit( my_moves, rand_move_idx );

    bool valid = othello_game_make_move( game, move );
    if( !valid ) Fail( "tried to make invalid move" );
  }
}
