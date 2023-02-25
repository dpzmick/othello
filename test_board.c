#include "unit.h"

#include "bitboard.h"
#include "util.h"

#include <assert.h>
#include <stdbool.h>

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

TEST( simple_get_all_moves )
{
  board_t board = new_board_from_str( "........"
                                      "....BBW."
                                      "...B...."
                                      "..W....."
                                      ".BBBW..."
                                      "....B..."
                                      "........"
                                      "........" );

  uint64_t all_moves = board_get_all_moves( &board, PLAYER_WHITE );

  uint64_t expected = parse_moves( ".....*.."
                                   "...*...."
                                   "........"
                                   "........"
                                   "*......."
                                   "*.*....."
                                   "....**.."
                                   "........" );

  CHECK_EQ( all_moves, expected );
}

TEST( basic_flips )
{

  {
    board_t board = new_board_from_str( "........"
                                        ".WBB...."
                                        "...W...."
                                        "...B...."
                                        "....BB.."
                                        "........"
                                        "...B...."
                                        "....B..." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 4, 1 );
    CHECK( valid );

    board_t expect = new_board_from_str( "........"
                                         ".WWWW..."
                                         "...W...."
                                         "...B...."
                                         "....BB.."
                                         "........"
                                         "...B...."
                                         "....B..." );

    CHECK( board_eq( &board, &expect ) );
  }

  {
    board_t board = new_board_from_str( "........"
                                        ".WBB...."
                                        "...W...."
                                        "...B...."
                                        "....BB.."
                                        "........"
                                        "...B...."
                                        "....B..." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 3, 4 );
    CHECK( valid );

    board_t expect = new_board_from_str( "........"
                                         ".WBB...."
                                         "...W...."
                                         "...W...."
                                         "...WBB.."
                                         "........"
                                         "...B...."
                                         "....B..." );

    CHECK( board_eq( &board, &expect ) );
  }

  {
    board_t board = new_board_from_str( "........"
                                        ".WBB...."
                                        "........"
                                        "........"
                                        "....BB.."
                                        "...B...."
                                        "...B...."
                                        "...W...." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 3, 4 );
    CHECK( valid );

    board_t expect = new_board_from_str( "........"
                                         ".WBB...."
                                         "........"
                                         "........"
                                         "...WBB.."
                                         "...W...."
                                         "...W...."
                                         "...W...." );

    CHECK( board_eq( &board, &expect ) );
  }

  {
    board_t board = new_board_from_str( "........"
                                        ".WBB...."
                                        "........"
                                        "........"
                                        "........"
                                        "...B...."
                                        "...B...."
                                        "...B...." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 3, 4 );
    CHECK( !valid );
  }

  {
    board_t board = new_board_from_str( "........"
                                        ".WBB...."
                                        "........"
                                        "........"
                                        "...BBW.."
                                        "...B...."
                                        "....B..."
                                        ".....B.." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 2, 4 );
    CHECK( valid );

    board_t expect = new_board_from_str( "........"
                                         ".WBB...."
                                         "........"
                                         "........"
                                         "..WWWW.."
                                         "...B...."
                                         "....B..."
                                         ".....B.." );

    CHECK( board_eq( &board, &expect ) );
  }

  {
    board_t board = new_board_from_str( ".W......"
                                        "..B....."
                                        "...B...."
                                        ".....BBB"
                                        "........"
                                        "........"
                                        "........"
                                        "........" );

    bool valid = board_make_move( &board, PLAYER_WHITE, 4, 3 );
    CHECK( valid );

    board_t expect = new_board_from_str( ".W......"
                                         "..W....."
                                         "...W...."
                                         "....WBBB"
                                         "........"
                                         "........"
                                         "........"
                                         "........" );

    CHECK( board_eq( &board, &expect ) );
  }

  {
    board_t board = new_board_from_str( "........"
                                        "........"
                                        "........"
                                        "........"
                                        "..W.W..."
                                        "...WB..."
                                        "........"
                                        ".....B.." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 4, 6 );
    CHECK( valid );

    board_t expect = new_board_from_str( "........"
                                         "........"
                                         "........"
                                         "........"
                                         "..W.W..."
                                         "...WW..."
                                         "....W..."
                                         ".....B.." );

    CHECK( board_eq( &board, &expect ) );
  }

  {
    board_t board = new_board_from_str( "..WWBBW."
                                        "..WBWB.."
                                        "..BBBBBB"
                                        "BBBBBBBB"
                                        "..WBBBBB"
                                        "...WBBBB"
                                        ".....BB."
                                        ".....B.." );

    bool valid = board_make_move( &board, PLAYER_WHITE, 4, 6 );
    CHECK( valid );

    board_t expect = new_board_from_str( "..WWBBW."
                                         "..WBWB.."
                                         "..BBWBBB"
                                         "BBBBWBBB"
                                         "..WBWBBB"
                                         "...WWBBB"
                                         "....WBB."
                                         ".....B.." );

    CHECK( board_eq( &board, &expect ) );
  }
}

TEST( test_board_moves )
{
    board_t b = new_board_from_str( "........"
                                    "........"
                                    "........"
                                    "........"
                                    "........"
                                    "........"
                                    "........"
                                    "........" );

    CHECK_EQ( board_total_moves(&b), 0 );
}

// FIXME finish up the wraparound tests. all the diagonals are missing them
