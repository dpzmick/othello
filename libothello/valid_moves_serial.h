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
