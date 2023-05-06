/* static inline void */
/* print_lanes( char const * which, uint64x4_t v ) */
/* { */
/*   for( size_t lane = 0; lane < 4; ++lane ) { */
/*     printf( "%s[%zu] = %lx\n", which, lane, v[lane] ); */
/*   } */
/* } */

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

  uint64x4_t owns = { own, own, own, own };
  uint64x4_t opps = { opp, opp, opp, opp };

  uint64_t flips = 0;

  { // -- start left

    uint64x4_t left_shifts = {
      UINT64_C(1), /* left */
      UINT64_C(8), /* up */
      UINT64_C(9), /* up-left */
      UINT64_C(7), /* up-right */
    };

    uint64x4_t left_masks = {
      UINT64_C(0xfefefefefefefefe), /* left */
      UINT64_C(0xffffffffffffffff), /* up */
      UINT64_C(0xfefefefefefefefe), /* up-left */
      UINT64_C(0x7f7f7f7f7f7f7f7f), /* up-right */
    };

    /* place to store temporary flips until they are fused into lane_flips */
    uint64x4_t pending_flips = {0, 0, 0, 0};

    /* A cursor move is valid when:
       - the appropriate direction mask has been applied
       - there is an opponent cell at the next location (also implies non-empty). */

    /* Start at out move location in all lanes */
    uint64x4_t cursors = { move, move, move, move };

    /* We can keep this lanes flips when the _next piece_ we'll visit is one of
       our own. We look ahead by one because the cursor update logic is zeroing
       the cursor when we hit any piece that is not opponent piece */

    uint64x4_t keep_flips = (cursors << left_shifts) & left_masks & owns;

    /* Apply the cursor update procedure */
    cursors = (cursors << left_shifts) & left_masks & opps;

    for( size_t iter = 0; iter < 8; ++iter ) {
      /* If cursor is still non-zero, accumulate path as flips */
      pending_flips |= cursors;

      /* Update keep_flips looking ahead by one position */
      keep_flips |= (cursors << left_shifts) & left_masks & owns;

      /* Update cursor following the rules */
      cursors = (cursors << left_shifts) & left_masks & opps;
    }

    /* Update flips with all the pending flips, but only if we have marked this
       lane as a keeper. */

    for( size_t lane = 0; lane < 4; ++lane ) {
      if( keep_flips[lane] ) flips |= pending_flips[lane];
    }
  } // -- end left

  { // -- start right
    uint64x4_t right_shifts = {
      UINT64_C(1), /* right */
      UINT64_C(8), /* down */
      UINT64_C(9), /* down-right */
      UINT64_C(7), /* down-left */
    };

    uint64x4_t right_masks = {
      UINT64_C(0x7f7f7f7f7f7f7f7f), /* right */
      UINT64_C(0xffffffffffffffff), /* down */
      UINT64_C(0x7f7f7f7f7f7f7f7f), /* down-right */
      UINT64_C(0xfefefefefefefefe), /* down-left */
    };

    /* place to store temporary flips until they are fused into lane_flips */
    uint64x4_t pending_flips = {0, 0, 0, 0};

    uint64x4_t cursors = { move, move, move, move };
    uint64x4_t keep_flips = (cursors >> right_shifts) & right_masks & owns;

    cursors = (cursors >> right_shifts) & right_masks & opps;

    for( size_t iter = 0; iter < 8; ++iter ) {
      pending_flips |= cursors;
      keep_flips |= (cursors >> right_shifts) & right_masks & owns;
      cursors = (cursors >> right_shifts) & right_masks & opps;
    }

    /* Update flips with all the pending flips, but only if we have marked this
       lane as a keeper. */

    for( size_t lane = 0; lane < 4; ++lane ) {
      if( keep_flips[lane] ) flips |= pending_flips[lane];
    }
  } // -- end right

  /* apply flips */
  own = own | flips | move;
  opp = opp & ~flips;

  *own_p = own;
  *opp_p = opp;

  game->curr_player = !game->curr_player;
  return true;
}
