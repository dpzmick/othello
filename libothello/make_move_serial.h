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

  /* We know this is a valid move.

     Opposing players pieces are flipped _if_ this move outflanks a run in any
     direction.

     We just need to check is if we outflank (hit an empty square before hitting
     out own) in each direction. As we're checking, we can track what pieces
     would have gotten flipped (perhaps we can compute those flips as part of
     the move calculation?) */

  uint64_t flips = 0;

  int64_t left_shifts[] = {
    INT64_C(1), /* left */
    INT64_C(8), /* up */
    INT64_C(9), /* up-left */
    INT64_C(7), /* up-right */
  };

  uint64_t left_masks[] = {
    UINT64_C(0xfefefefefefefefe), /* left */
    UINT64_C(0xffffffffffffffff), /* up */
    UINT64_C(0xfefefefefefefefe), /* up-left */
    UINT64_C(0x7f7f7f7f7f7f7f7f), /* up-right */
  };

  for( size_t dir = 0; dir < ARRAY_SIZE( left_shifts ); ++dir ) {
    int64_t  shift_amount = left_shifts[dir];
    uint64_t shift_mask   = left_masks[dir];

    uint64_t dir_flips = 0;

    /* start at the move location shifted in direction */
    uint64_t cursor = (move << shift_amount) & shift_mask;

    while( true ) {
      /* if cursor is expired (went off edge), exit */
      if( cursor==0 ) break;

      /* if we hit an empty piece, exit. not outflanking here */
      if( (cursor&empty) != 0 ) break;

      /* if we hit our own piece, we've finished the line. Add to flips! */
      if( (cursor&own) != 0 ) {
        flips |= dir_flips;
        break;
      }

      dir_flips |= cursor;
      cursor = (cursor << shift_amount) & shift_mask;
    }
  }

  uint64_t right_shifts[] = {
    UINT64_C(1), /* right */
    UINT64_C(8), /* down */
    UINT64_C(9), /* down-right */
    UINT64_C(7), /* down-left */
  };

  uint64_t right_masks[] = {
    UINT64_C(0x7f7f7f7f7f7f7f7f), /* right */
    UINT64_C(0xffffffffffffffff), /* down */
    UINT64_C(0x7f7f7f7f7f7f7f7f), /* down-right */
    UINT64_C(0xfefefefefefefefe), /* down-left */
  };

  for( size_t dir = 0; dir < ARRAY_SIZE( left_shifts ); ++dir ) {
    uint64_t shift_amount = right_shifts[dir];
    uint64_t shift_mask   = right_masks[dir];

    uint64_t dir_flips = 0;

    /* start at the move location shifted in direction */
    uint64_t cursor = (move >> shift_amount) & shift_mask;

    while( true ) {
      /* if cursor is expired (went off edge), exit */
      if( cursor==0 ) break;

      /* if we hit an empty piece, exit. not outflanking here */
      if( (cursor&empty) != 0 ) break;

      /* if we hit our own piece, we've finished the line. Add to flips! */
      if( (cursor&own) != 0 ) {
        flips |= dir_flips;
        break;
      }

      dir_flips |= cursor;
      cursor = (cursor >> shift_amount) & shift_mask;
    }
  }

  /* apply flips */
  own = own | flips | move;
  opp = opp & ~flips;

  *own_p = own;
  *opp_p = opp;

  game->curr_player = !game->curr_player;
  return true;
}
