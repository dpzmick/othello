#include "unit.h"

#include "bitboard.h"

#include <assert.h>
#include <stdbool.h>

uint64_t parse_moves( char const * moves )
{
  uint64_t ret = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      switch( *moves++ ) {
        case '*':
          ret |= BIT_MASK(x,y);
          break;
      }
    }
  }

  return ret;
}

TEST( sanity )
{
  CHECK_EQ( 1, 1 );
}

TEST( test_parse )
{
  board_t board = new_board_from_str( "W..B..W."
                                      "B..W..B."
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      ".......W" );

  uint64_t white = bitboard_from_rows((uint8_t[8]){
    0b10000010,
    0b00010000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000001 });

  uint64_t black = bitboard_from_rows((uint8_t[8]){
    0b00010000,
    0b10000010,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000 });

  // board_print( &board );

  CHECK_EQ( board.white, white );
  CHECK_EQ( board.black, black );
}

TEST( right_moves )
{
  board_t board = new_board_from_str( "WBBBBBB." // should get edge
                                      "........"
                                      "WBBBBBBW" // no valid move here
                                      ".BBW...." // no valid move here
                                      "....WBB." // should get edge
                                      "........"
                                      "..WBB..." // should get middle
                                      "........" );

  uint64_t expected = parse_moves( ".......*"
                                   "........"
                                   "........"
                                   "........"
                                   ".......*"
                                   "........"
                                   ".....*.."
                                   "........" );

  uint64_t moves = board_right_moves( board.white, board.black );
  assert( moves==expected );
}

TEST( right_moves_wrap_around )
{
  /* make sure edges are masked off correctly */
  board_t board = new_board_from_str( "WBBBBBBB"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........" );

  uint64_t moves = board_right_moves( board.white, board.black );
  assert( moves==0 );
}

TEST( left_moves )
{
  board_t board = new_board_from_str( ".BBBBBBW" // should get edge
                                      "........"
                                      "WBBBBBBW" // no valid move here
                                      ".WBB...." // no valid move here
                                      ".BBW...." // should get edge
                                      "........"
                                      "..BBW..." // should get middle
                                      "........" );

  uint64_t expected = parse_moves( "*......."
                                   "........"
                                   "........"
                                   "........"
                                   "*......."
                                   "........"
                                   ".*......"
                                   "........" );

  uint64_t moves = board_left_moves( board.white, board.black );
  assert( moves==expected );
}

TEST( left_moves_wrap_around )
{
  /* make sure edges are masked off correctly */
  board_t board = new_board_from_str( "........"
                                      "BBBBBBBW"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........" );

  uint64_t moves = board_left_moves( board.white, board.black );
  assert( moves==0 );
}

TEST( up_moves )
{
  board_t board = new_board_from_str( "...B...."
                                      "B..B...."
                                      "W..W...."
                                      "........"
                                      ".....B.."
                                      ".....B.."
                                      ".....B.."
                                      ".....W.." );

  uint64_t expected = parse_moves( "*......."
                                   "........"
                                   "........"
                                   ".....*.."
                                   "........"
                                   "........"
                                   "........"
                                   "........" );

  uint64_t moves = board_up_moves( board.white, board.black );
  assert( moves==expected );
}

TEST( up_moves_wrap_around )
{
  // not possible for up moves becuse shifting always fills in with zeros
  // there is no wraparound to consider
}

TEST( down_moves )
{
  board_t board = new_board_from_str( "...W...."
                                      "W..B...."
                                      "B..B...."
                                      "........"
                                      ".....W.."
                                      ".....B.."
                                      ".....B.."
                                      "........" );

  uint64_t expected = parse_moves( "........"
                                   "........"
                                   "........"
                                   "*..*...."
                                   "........"
                                   "........"
                                   "........"
                                   ".....*.." );

  uint64_t moves = board_down_moves( board.white, board.black );
  assert( moves==expected );
}

TEST( down_moves_wrap_around )
{
  /* make sure edges are masked off correctly */
  board_t board = new_board_from_str( "..B....."
                                      "..B....."
                                      "..W....."
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........" );

  uint64_t moves = board_down_moves( board.white, board.black );
  assert( moves==0 );
}

TEST( down_left_moves )
{
  board_t board = new_board_from_str( "W...W..."
                                      ".B...B.."
                                      "..B....."
                                      "........"
                                      "........"
                                      "W....W.."
                                      ".B....B."
                                      "..B....." );

  uint64_t expected = parse_moves( "........"
                                   "........"
                                   "......*."
                                   "...*...."
                                   "........"
                                   "........"
                                   "........"
                                   ".......*" );

  uint64_t moves = board_down_left_moves( board.white, board.black );
  CHECK_EQ( moves, expected );

  board = new_board_from_str( "W......."
                              ".B......"
                              "..B....."
                              "...B...."
                              "....B..."
                              ".....B.."
                              "......B."
                              "........" );

  expected = parse_moves( "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          ".......*" );

  moves = board_down_left_moves( board.white, board.black );
  CHECK_EQ( moves, expected );
}

TEST( down_right_moves )
{
  board_t board = new_board_from_str( "....W..."
                                      "...B..W."
                                      "..B..B.."
                                      "........"
                                      ".......W"
                                      "......B."
                                      ".....B.."
                                      "........" );

  uint64_t expected = parse_moves( "........"
                                   "........"
                                   "........"
                                   ".*..*..."
                                   "........"
                                   "........"
                                   "........"
                                   "....*..." );

  uint64_t moves = board_down_right_moves( board.white, board.black );
  CHECK_EQ( moves, expected );

  board = new_board_from_str( ".......W"
                              "......B."
                              ".....B.."
                              "....B..."
                              "...B...."
                              "..B....."
                              ".B......"
                              "........" );

  expected = parse_moves( "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "*......." );

  moves = board_down_right_moves( board.white, board.black );
  CHECK_EQ( moves, expected );
}

TEST( up_left_moves )
{
  board_t board = new_board_from_str( "........"
                                      "..B....."
                                      "...B...."
                                      "....W..."
                                      "........"
                                      "B...B..."
                                      ".B...W.."
                                      "..W....." );

  uint64_t expected = parse_moves( ".*......"
                                   "........"
                                   "........"
                                   "........"
                                   "...*...."
                                   "........"
                                   "........"
                                   "........" );

  uint64_t moves = board_up_left_moves( board.white, board.black );
  CHECK_EQ( moves, expected );

  board = new_board_from_str( "........"
                              ".B......"
                              "..B....."
                              "...B...."
                              "....B..."
                              ".....B.."
                              "......B."
                              ".......W" );

  expected = parse_moves( "*......."
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........" );

  moves = board_up_left_moves( board.white, board.black );
  CHECK_EQ( moves, expected );
}

TEST( up_right_moves )
{
  board_t board = new_board_from_str( "........"
                                      "....B..."
                                      "...B...B"
                                      "..W...W."
                                      "........"
                                      "....B..."
                                      "...B...B"
                                      "..W...W." );

  uint64_t expected = parse_moves( ".....*.."
                                   "........"
                                   "........"
                                   "........"
                                   ".....*.."
                                   "........"
                                   "........"
                                   "........" );

  uint64_t moves = board_up_right_moves( board.white, board.black );
  CHECK_EQ( moves, expected );

  board = new_board_from_str( "........"
                              "......B."
                              ".....B.."
                              "....B..."
                              "...B...."
                              "..B....."
                              ".B......"
                              "W......." );

  expected = parse_moves( ".......*"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........"
                          "........" );

  moves = board_up_right_moves( board.white, board.black );
  CHECK_EQ( moves, expected );
}

int main()
{
  return unit_test_run_all();
}

// FIXME finish up the wraparound tests. all the diagonals are missing them
