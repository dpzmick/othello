#include "unit.h"

#include "util.h"
#include "game_tree.h"

#include <stdbool.h>
#include <stdlib.h>

TEST( sanity )
{
  CHECK_EQ( 1, 1 );
}

TEST( test_board_lookup )
{
  board_t board = new_board_from_str( "W..B..W."
                                      "B..W..B."
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      "........"
                                      ".......W" );
  size_t n = 32;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  {
    node_t * node = game_table_get( gt, board );
    CHECK( board_eq( &board, &node->board ) );
    CHECK_EQ( node->win_cnt, 0 );
    CHECK_EQ( node->game_cnt, 0 );

    node->win_cnt = 10;
  }

  {
    node_t * node = game_table_get( gt, board );
    CHECK( board_eq( &board, &node->board ) );
    CHECK_EQ( node->win_cnt, 10 );
    CHECK_EQ( node->game_cnt, 0 );
  }

  //printf( "gets: %zu, loops: %zu\n", gt->n_gets, gt->n_loops );

  free( mem );
}

TEST( test_board_abuse )
{
  // generate random boards and do more lookups than space in the board
  //
  // note that the random boards access pattern is extremely unrealistic, boards
  // in the real game will only move "forward"
  //
  // this will quickly get stuck, predictably, with all boards having many
  // pieces, so new boards with less pieces added cannot be played.

  size_t n = 32;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  size_t worked = 0;
  for( size_t i = 0; true; ++i ) {
    board_t board;
    board_init_random( &board, i );

    node_t * node = game_table_get( gt, board );
    if( !node ) break; // keep going until we can't generate any new boards that fit

    CHECK( board_eq( &board, &node->board ) );
    worked += 1;
  }

  CHECK( worked >= 32 );

  /* printf( "worked: %zu, gets: %zu, loops: %zu\n", worked, gt->n_gets, gt->n_loops ); */
  /* printf( "avg loops per get: %f\n", (double)gt->n_loops / (double)gt->n_gets ); */

  free( mem );
}

TEST( test_board_replacing )
{
  size_t n = 4;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  size_t worked = 0;
  for( size_t i = 1; i<=64; ++i ) { // start at 1
    board_t board;
    board_init_random_n_set( &board, 0xcafecafebabebabe, i );

    /* printf("\n"); */
    /* board_print( &board ); */

    node_t * node = game_table_get( gt, board );
    REQUIRE( node );
    CHECK( board_eq( &board, &node->board ) );
    worked += 1;
  }

  CHECK( worked==64 );

  /* printf( "worked: %zu, gets: %zu, loops: %zu\n", worked, gt->n_gets, gt->n_loops ); */
  /* printf( "avg loops per get: %f\n", (double)gt->n_loops / (double)gt->n_gets ); */

  free( mem );
}

TEST( test_board_abuse_replacing )
{
  size_t n = 32;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  size_t worked = 0;
  for( size_t i = 1; i<=64; ++i ) { // start at 1
    // generate n different boards to insert
    for( size_t j = 1; j <= n; ++j ) { // start at 1 to avoid blowing away seed
      board_t board;
      board_init_random_n_set( &board, 0xcafecafebabebabe*j, i );

      /* printf("\n"); */
      /* board_print( &board ); */

      node_t * node = game_table_get( gt, board );
      REQUIRE( node );
      CHECK( board_eq( &board, &node->board ) );
      worked += 1;
    }
  }

  CHECK( worked == 64*n );

  // when I ran this I got average of 3.9 loops per lookup with 32 slots, which
  // is decent enough.

  /* printf( "worked: %zu, gets: %zu, loops: %zu\n", worked, gt->n_gets, gt->n_loops ); */
  /* printf( "avg loops per get: %f\n", (double)gt->n_loops / (double)gt->n_gets ); */

  free( mem );
}

#if 0
TEST( test_board_abuse_benchmark )
{
  size_t n = 8192;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  // allocate a ton of memory and save all of the boards to avoid
  // doing the expensive setup in the loop

  board_t boards[64][n]; // VLA!

  for( size_t i = 1; i<=64; ++i ) {
    for( size_t j = 1; j <= n; ++j ) {
      board_init_random_n_set( &boards[i-1][j-1], 0xcafecafebabebabe*j, i );
    }
  }

  uint64_t st  = wallclock();
  size_t   cnt = 0;

  for( size_t i = 1; i<=64; ++i ) { // start at 1
    for( size_t j = 1; j <= n; ++j ) { // start at 1 to avoid blowing away seed
      node_t * node = game_table_get( gt, boards[i-1][j-1] );
      cnt += !!node;
    }
  }

  uint64_t ed = wallclock();

  uint64_t lookups         = gt->n_gets;
  double   sec             = ((double)(ed-st))/1e9;
  double   lookups_per_sec = ((double)lookups)/sec;
  double   ns_per_lookup   = ((double)(ed-st))/((double)lookups);

  // on M1, _unoptimized_ got 230ns per lookup.
  //
  // but doing avg of 53 loops per get. One cache line is 512 so that's sort of
  // okay, but still not great. It may be valuable to robinhood the tombstoned
  // elements forward (but depends on access pattern).
  //
  // that said, the rest of the engine is not going to be able to run 4million
  // games per second, so this is probably not the slow part (and 4 million per
  // second is comparable to some of the "fastest hash tables" out in the wild)
  //
  // optimized got 11.8 million lookups per second

  printf( "lookups: %llu, Successfull %zu, total elapsed %0.2f sec\n", lookups, cnt, sec );
  printf( "Lookups per sec: %0.2f\n", lookups_per_sec );
  printf( "ns per lookup: %0.2f\n", ns_per_lookup );
  printf( "loops: %zu, avg loops per get %0.2f\n", gt->n_loops, (double)gt->n_loops/(double)gt->n_gets );

  free( mem );
}
#endif

TEST( pick_next_move )
{
  size_t n = 8192;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  board_t board;
  board_init( &board );

  uint64_t st  = wallclock();

  uint64_t move = pick_next_move( gt, board, PLAYER_WHITE );
  CHECK( move != MOVE_PASS );

  uint64_t ed = wallclock();

  double sec = ((double)(ed-st))/1e9;

  printf("picked move %llx in %0.2f sec\n", move, sec );
}

TEST( pick_next_move_against_random_player )
{
  size_t n = 8192;
  void * mem = malloc( game_table_size( n ) );
  REQUIRE( mem );

  game_table_t * gt = game_table_new( mem, n );
  REQUIRE( gt );

  uint64_t st = wallclock();

  board_t board;
  board_init( &board );

  // white goes first

  player_t winner;
  while( 1 ) {
    if( board_is_game_over( &board, &winner ) ) break;
    board_print( &board );

    uint64_t moves = board_get_all_moves( &board, PLAYER_WHITE );
    if( moves ) {
      uint64_t n_moves       = (uint64_t)__builtin_popcountll( moves );
      uint64_t rand_move_idx = hash_u64( st*board.white*board.black ) % n_moves; // FIXME modulo bad

      uint64_t x,y;
      extract_move( moves, rand_move_idx, &x, &y );

      bool ret = board_make_move( &board, PLAYER_WHITE, x, y );
      CHECK( ret );
    }

    board_print( &board );

    uint64_t move = pick_next_move( gt, board, PLAYER_BLACK );
    if( move != MOVE_PASS ) {
      // FIXME gross
      for( size_t x = 0; x < 8; ++x ) {
        for( size_t y = 0; y < 8; ++y ) {
          if( move & BIT_MASK(x,y) ) {
            bool ret = board_make_move( &board, PLAYER_BLACK, x, y );
            CHECK( ret );
          }
        }
      }
    }
  }

  uint64_t ed = wallclock();

  double sec = ((double)(ed-st))/1e9;

  printf("played game in %0.2f sec, winner is %d. Final board:\n", sec, winner );
  board_print( &board );
}
