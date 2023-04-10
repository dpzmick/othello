#include "../libothello/othello.h"
#include "../libunit/unit.h"

#include <stdbool.h>
#include <stdint.h>

static uint64_t
bitboard_from_rows( uint8_t rows[8] )
{
  uint64_t ret = 0;
  for( size_t row = 0; row < 8; ++row ) {
    uint64_t shift = 64 - 8*(row+1);
    ret |= ((uint64_t)rows[row]) << shift;
  }
  return ret;
}

static uint64_t
parse_moves( char const * moves )
{
  uint64_t ret = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      switch( *moves++ ) {
        case '*':
          ret |= othello_bit_mask(x,y);
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
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "W..B..W."
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

  CHECK_EQ( game->white, white );
  CHECK_EQ( game->black, black );
}

TEST( right_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "WBBBBBB." // should hit edge
                              "........"
                              "WBBBBBBW" // no valid move here
                              "........" // no valid move here
                              "....WBB." // should get edge move
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( right_moves_wrap_around )
{
  /* make sure edges are masked off correctly */
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "WBBBBBBB"
                              "........"
                              "........"
                              "........"
                              "........"
                              "........"
                              "........"
                              "........" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

TEST( left_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              ".BBBBBBW" // should get edge
                              "........"
                              "WBBBBBBW" // no valid move here
                              "........"
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( left_moves_wrap_around )
{
  /* make sure edges are masked off correctly */
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "BBBBBBBW"
                              "........"
                              "........"
                              "........"
                              "........"
                              "........"
                              "........"
                              "........" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

TEST( up_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "...B...."
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( up_moves_wrap_around )
{
  /* not possible for up moves becuse shifting always fills in with zeros
     there is no wraparound to consider */
}

TEST( down_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "...W...."
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( down_moves_wrap_around )
{
  /* make sure edges are masked off correctly */
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "..B....."
                              "..B....."
                              "..W....."
                              "........"
                              "........"
                              "........"
                              "........"
                              "........" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

TEST( down_right_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "....W..."
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );

  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              ".......W"
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

  moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( down_right_moves_wraparound )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              ".......W"
                              "......B."
                              ".....B.."
                              "....B..."
                              "...B...."
                              "..B....."
                              ".B......"
                              "B......." );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

TEST( down_left_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "W...W..."
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );

  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "W......."
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

  moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( down_left_moves_wrap_around )
{
  /* We are shifting right, so make sure right edges get masked off */
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "W......."
                              ".B......"
                              "..B....."
                              "...B...."
                              "....B..."
                              ".....B.."
                              "......B."
                              ".......B" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

TEST( up_right_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "........"
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );

  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "........"
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

  moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( up_right_moves_wraparound )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              ".......B"
                              "......B."
                              ".....B.."
                              "....B..."
                              "...B...."
                              "..B....."
                              ".B......"
                              "W......." );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

TEST( up_left_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "........"
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

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );

  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "........"
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

  moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( up_left_moves_wraparound )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "B......."
                              ".B......"
                              "..B....."
                              "...B...."
                              "....B..."
                              ".....B.."
                              "......B."
                              ".......W" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, 0 );
}

/* Try getting a mix of moves at same time */

TEST( simple_get_all_moves )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_WHITE,
                              "........"
                              "....BBW."
                              "...B...."
                              "..W....."
                              ".BBBW..."
                              "....B..."
                              "........"
                                      "........" );

  uint64_t expected = parse_moves( ".....*.."
                                   "...*...."
                                   "........"
                                   "........"
                                   "*......."
                                   "*.*....."
                                   "....**.."
                                   "........" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

TEST( start_of_game_all_moves )
{
  othello_game_t game[1];
  othello_game_init( game );

  /* black goes first */
  uint64_t expected = parse_moves( "........"
                                   "........"
                                   "...*...."
                                   "..*....."
                                   ".....*.."
                                   "....*..."
                                   "........"
                                   "........" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );
}

/* was a bug */
TEST( weird_get_all_moves_1 )
{
  othello_game_t game[1];
  othello_game_init_from_str( game,
                              OTHELLO_BIT_BLACK,
                              "WWWWWWWB"
                              "WWWBBBBB"
                              "WBWWBBWB"
                              "WBBWBBWB"
                              "WBWBWBWB"
                              "WBWBBWW."
                              "WBBWWWWW"
                              "WWWWWWWW" );

  uint64_t expected = parse_moves( "........"
                                   "........"
                                   "........"
                                   "........"
                                   "........"
                                   ".......*"
                                   "........"
                                   "........" );

  uint64_t moves = othello_game_all_valid_moves( game );
  CHECK_EQ( moves, expected );

}

TEST( basic_flips )
{

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "........"
                                ".WBB...."
                                "...W...."
                                "...B...."
                                "....BB.."
                                "........"
                                "...B...."
                                "....B..." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 4, 1 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                "........"
                                ".WWWW..."
                                "...W...."
                                "...B...."
                                "....BB.."
                                "........"
                                "...B...."
                                "....B..." );

    CHECK( othello_game_eq( game, expect ) );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "........"
                                ".WBB...."
                                "...W...."
                                "...B...."
                                "....BB.."
                                "........"
                                "...B...."
                                "....B..." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 3, 4 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                "........"
                                ".WBB...."
                                "...W...."
                                "...W...."
                                "...WBB.."
                                "........"
                                "...B...."
                                "....B..." );

    CHECK( othello_game_eq( game, expect ) );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "........"
                                ".WBB...."
                                "........"
                                "........"
                                "....BB.."
                                "...B...."
                                "...B...."
                                "...W...." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 3, 4 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                "........"
                                ".WBB...."
                                "........"
                                "........"
                                "...WBB.."
                                "...W...."
                                "...W...."
                                "...W...." );

    CHECK( othello_game_eq( game, expect ) );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "........"
                                ".WBB...."
                                "........"
                                "........"
                                "........"
                                "...B...."
                                "...B...."
                                "...B...." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 3, 4 ) );
    CHECK( !valid );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "........"
                                ".WBB...."
                                "........"
                                "........"
                                "...BBW.."
                                "...B...."
                                "....B..."
                                ".....B.." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 2, 4 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                "........"
                                ".WBB...."
                                "........"
                                "........"
                                "..WWWW.."
                                "...B...."
                                "....B..."
                                ".....B.." );

    CHECK( othello_game_eq( game, expect ) );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                ".W......"
                                "..B....."
                                "...B...."
                                ".....BBB"
                                "........"
                                "........"
                                "........"
                                "........" );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 4, 3 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                ".W......"
                                "..W....."
                                "...W...."
                                "....WBBB"
                                "........"
                                "........"
                                "........"
                                "........" );

    CHECK( othello_game_eq( game, expect ) );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "........"
                                "........"
                                "........"
                                "........"
                                "..W.W..."
                                "...WB..."
                                "........"
                                ".....B.." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 4, 6 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                "........"
                                "........"
                                "........"
                                "........"
                                "..W.W..."
                                "...WW..."
                                "....W..."
                                ".....B.." );

    CHECK( othello_game_eq( game, expect ) );
  }

  {
    othello_game_t game[1];
    othello_game_init_from_str( game,
                                OTHELLO_BIT_WHITE, // white to make next move
                                "..WWBBW."
                                "..WBWB.."
                                "..BBBBBB"
                                "BBBBBBBB"
                                "..WBBBBB"
                                "...WBBBB"
                                ".....BB."
                                ".....B.." );

    uint8_t winner;
    othello_move_ctx_t ctx[1];
    bool valid = othello_game_start_move( game, ctx, &winner );
    REQUIRE( valid );

    valid = othello_game_make_move( game, ctx, othello_bit_mask( 4, 6 ) );
    CHECK( valid );

    othello_game_t expect[1];
    othello_game_init_from_str( expect,
                                OTHELLO_BIT_BLACK,
                                "..WWBBW."
                                "..WBWB.."
                                "..BBWBBB"
                                "BBBBWBBB"
                                "..WBWBBB"
                                "...WWBBB"
                                "....WBB."
                                ".....B.." );

    CHECK( othello_game_eq( game, expect ) );
  }
}

TEST( basic_random_play )
{
  /* check that this doesn't segfault or assert, and that it actually terminates */
  othello_game_t game[1];
  othello_game_init( game );

  uint8_t result = othello_game_random_playout( game, 0xcafebabedeadbeefUL );
  CHECK(
    result==OTHELLO_BIT_WHITE
    || result==OTHELLO_BIT_BLACK
    || result==OTHELLO_GAME_TIED) ;
}
