#include "unit.h"

#include "util.h"
#include "game_tree.h"

#include <assert.h>
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
    CHECK_EQ( node->wins, 0 );
    CHECK_EQ( node->game_cnt, 0 );

    node->wins = 10;
  }

  {
    node_t * node = game_table_get( gt, board );
    CHECK( board_eq( &board, &node->board ) );
    CHECK_EQ( node->wins, 10 );
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

  // on M1, got 230ns per lookup.
  //
  // but doing avg of 53 loops per get. One cache line is 512 so that's sort of
  // okay, but still not great. It may be valuable to robinhood the tombstoned
  // elements forward (but depends on access pattern).
  //
  // that said, the rest of the engine is not going to be able to run 4million
  // games per second, so this is probably not the slow part (and 4 million per
  // second is comparable to some of the "fastest hash tables" out in the wild)

  printf( "lookups: %llu, Successfull %zu, total elapsed %0.2f sec\n", lookups, cnt, sec );
  printf( "Lookups per sec: %0.2f\n", lookups_per_sec );
  printf( "ns per lookup: %0.2f\n", ns_per_lookup );
  printf( "loops: %zu, avg loops per get %0.2f\n", gt->n_loops, (double)gt->n_loops/(double)gt->n_gets );

  free( mem );
}
#endif
