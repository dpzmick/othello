#include "othello.h"
#include "../libcommon/hash.h"
#include "../libcommon/common.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

//#include "valid_moves_serial.h"

#ifdef __ARM_NEON__
#include "valid_moves_vectorized_arm.h"
#else
#include "valid_moves_vectorized_intel.h" // FIXME realy should be "generic"
#endif

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
othello_game_start_move( othello_game_t const * game,
                         othello_move_ctx_t *   ctx,
                         uint8_t *              out_winner )
{
  uint8_t curr_player = game->curr_player;

  uint64_t own_moves = _all_valid_moves( game, curr_player );
  uint64_t opp_moves = _all_valid_moves( game, !curr_player );

  uint64_t n_own_moves = (uint64_t)__builtin_popcountll( own_moves );
  uint64_t n_opp_moves = (uint64_t)__builtin_popcountll( opp_moves );

  if( n_own_moves==0 && n_opp_moves==0 ) {
    /* game is over */
    uint64_t white_stones = (uint64_t)__builtin_popcountll( game->white );
    uint64_t black_stones = (uint64_t)__builtin_popcountll( game->black );

    if( white_stones > black_stones )      *out_winner = OTHELLO_BIT_WHITE;
    else if( black_stones > white_stones ) *out_winner = OTHELLO_BIT_BLACK;
    else                                   *out_winner = OTHELLO_GAME_TIED;

    return false;
  }
  else {
    /* game continues */
    ctx->own_moves   = own_moves;
    ctx->n_own_moves = n_own_moves;
    ctx->opp_moves   = opp_moves;
    ctx->n_opp_moves = n_opp_moves;
    return true;
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
othello_game_make_move( othello_game_t *           game,
                        othello_move_ctx_t const * ctx,
                        uint64_t                   move )
{
  if( move == OTHELLO_MOVE_PASS ) {
    game->curr_player = !game->curr_player;
    return true;
  }

  if( 1 != __builtin_popcountll( move ) ) {
    return false;
  }

  uint8_t    player    = game->curr_player;
  uint64_t * own_p     = player==OTHELLO_BIT_WHITE ? &game->white : &game->black;
  uint64_t * opp_p     = player==OTHELLO_BIT_WHITE ? &game->black : &game->white;
  uint64_t   own_moves = ctx->own_moves;

  if( (move & own_moves) == 0 ) {
    return false;
  }

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
    uint8_t            winner;
    othello_move_ctx_t ctx[1];
    if( !othello_game_start_move( game, ctx, &winner ) ) {
      return winner;
    }

    uint64_t own_moves   = ctx->own_moves;
    uint64_t n_own_moves = ctx->n_own_moves;

    if( n_own_moves==0 ) {
      bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
      if( !valid ) Fail( "tried to make invalid move" );
      continue;
    }

    // pick a move at random.
    // FIXME modulo isn't great
    uint64_t rand_move_idx = hash_u64( seed+cnt ) % n_own_moves;
    uint64_t move          = keep_ith_set_bit( own_moves, rand_move_idx );

    bool valid = othello_game_make_move( game, ctx, move );
    if( !valid ) Fail( "tried to make invalid move" );
  }
}
