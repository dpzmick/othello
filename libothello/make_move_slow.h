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
