#pragma once

#include "common.h"
#include "bitboard.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/* #define XXH_INLINE_ALL */
/* #define XXH_NO_STDLIB */
/* #include <xxhash.h> */

uint64_t XXH3_64bits( void * buf, size_t sz )
{
  (void)buf;
  (void)sz;
  return 1;
}

#define LIKELY(c)   __builtin_expect(c, 1)
#define UNLIKELY(c) __builtin_expect(c, 0)

static inline uint64_t
next_pow2( uint64_t x )
{
  // count leading zeros.. 0000001.... returns 6
  if( x<2 ) return 2; // manage overflow and UB w/ clzll(0)
  uint64_t first_one = 64ul - (uint64_t)__builtin_clzll( x-1 ); // -1 in case number is already a power of two
  return 1ul << first_one;
}

// store game tree in a hash table
//
// looking up the current game state (if we've seen it before) is easy
//
// we can invalidate slots implicitly by checking if their state is still
// reachable. If it's not, we can reuse the slot
//
// "traversal" is modifying the game state and looking up the next node
//
// there are many possible next nodes (any possible move)
// we need to know _which_ next moves we've already tried? FIXME do we?
// what is more expensive? recomputing future states and hashing or just storing next pointers
//
// hash table is also a nice way to statically allocate, but a pool+free list
// would be fine for that too
// FIXME how to size it?

typedef struct {
  board_t  board;               // board stored in this node
  uint64_t win_cnt;             // number of simulated wins from this position
  uint64_t game_cnt;            // number of simulated games played from this node
} node_t;

// 16 KiB of D-cache means we can fit 512 entries of this table into cache
// whole ram is 16 MiB

typedef struct {
  size_t n_nodes;
  size_t mask;
  size_t n_gets;                // stats
  size_t n_loops;               // stats
  node_t nodes[];
} game_table_t;

// we can't really estimate the load factor and rehash "sometimes" since cells
// become tombstoned automatically.
//
// this structure needs to be used for a long time so we will need to deal with
// tombstones increasing the load factor too much... so, we'll do cleanup as we
// go

static inline size_t
game_table_size( size_t n_nodes )
{
  n_nodes = next_pow2( n_nodes );
  return sizeof(game_table_t) + n_nodes*sizeof(node_t);
}

static inline game_table_t *
game_table_new( void * mem,
                size_t n_nodes )
{
  game_table_t * ret = mem;
  ret->n_nodes = next_pow2( n_nodes );
  ret->mask    = ret->n_nodes - 1;
  ret->n_gets  = 0;
  ret->n_loops = 0;

  memset( ret->nodes, 0, sizeof(node_t)*ret->n_nodes );

  return ret;
}

// node is valid until the next call to get

static inline node_t *
game_table_get_advanced( game_table_t * gt,
                         board_t        board,
                         bool           allow_insert )
{
  size_t   board_stones = board_total_stones( &board );
  size_t   mask        = gt->mask;
  node_t * nodes       = gt->nodes;
  uint64_t hash        = XXH3_64bits( &board, sizeof(board) );
  size_t   slot        = hash & mask;
  node_t * ret         = NULL;
  size_t   n_loops     = 0;

  while( 1 ) {
    node_t * node              = &nodes[slot];
    board_t  node_board        = node->board;
    size_t   node_board_stones = board_total_stones( &node_board );

    n_loops += 1;

    // found the matching cell
    if( LIKELY( board_eq( &node_board, &board ) ) ) {
      ret = node;
      goto done;
    }

    // boards always evolve by adding new stones
    // a cell is "empty" if it contains an earlier game state that we no longer care about
    if( allow_insert && board_stones > node_board_stones ) {
      // reset node and use it
      node->board    = board;
      node->win_cnt  = 0;
      node->game_cnt = 0;
      ret            = node;
      goto done;
    }

    slot = (slot+1) & mask;

    if( UNLIKELY( slot == (hash & mask) ) ) { // out of space, we wrapped around
      ret = NULL;
      goto done;
    }
  }

done:
  gt->n_gets += 1;
  gt->n_loops += n_loops;
  return ret;
}

static inline node_t *
game_table_get( game_table_t * gt,
                board_t        board )
{
  // hopefully the `true` constant can get inlined here
  return game_table_get_advanced( gt, board, true );
}

// ---------------------------------------------------------------------
#ifndef TARGET_PLAYDATE
#include <stdio.h>

static inline void
print_tree( game_table_t * gt,
            board_t        board,
            player_t       next_player,
            size_t         depth )
{
  node_t * n = game_table_get_advanced( gt, board, false );
  if( !n ) return;

  for( size_t i = 0; i < depth; ++i ) printf( " " );
  printf( "[%zu] white: 0x%llx, black: 0x%llx. wins: %llu, games: %llu\n",
          depth,
          board.white, board.black,
          n->win_cnt, n->game_cnt );

  uint64_t moves   = board_get_all_moves( &board, next_player );
  uint64_t n_moves = (uint64_t)__builtin_popcountll( moves );

  for( size_t move = 0; move < n_moves; ++move ) {
    uint64_t mx, my;
    extract_move( moves, move, &mx, &my );

    /* for( size_t i = 0; i < depth; ++i ) printf( " " ); */
    /* printf( "Could move at (%llu, %llu)\n", mx, my ); */

    board_t child = board;
    board_make_move( &child, next_player, mx, my );

    print_tree( gt, child, !next_player, depth+1 );
  }
}
#endif

static inline void
select_from_children( game_table_t * gt,
                      node_t         curr,
                      player_t       next_player,
                      uint64_t       moves,
                      size_t         n_moves,
                      uint64_t *     out_mx,
                      uint64_t *     out_my )
{
  uint64_t picked_move_x = (size_t)-1;
  uint64_t picked_move_y = (size_t)-1;
  float best_criteria = -1.0f; // all computed criteria are positive

  /* printf("looking for moves with %zu options\n", n_moves); */
  for( size_t move = 0; move < n_moves; ++move ) {
    uint64_t mx, my;
    extract_move( moves, move, &mx, &my );

    board_t upd_board = curr.board;
    bool ret = board_make_move( &upd_board, next_player, mx, my );
    assert( ret );

    node_t * move_node = game_table_get( gt, upd_board );
    assert( move_node ); // table isn't big enough

    // if we find a node we've never played at before, try it
    if( move_node->game_cnt==0 ) {
      picked_move_x = mx;
      picked_move_y = my;
      best_criteria = INFINITY;
      break;
    }

    float criteria = 0.0f;
    if( move_node->game_cnt > 0 ) {
      criteria += (float)move_node->win_cnt / (float)move_node->game_cnt;
    }

    if( curr.game_cnt > 0 && move_node->game_cnt > 0 ) {
      criteria += sqrtf(2.0) * sqrtf(log2f(curr.game_cnt) / (float)move_node->game_cnt);
    }

    if( criteria > best_criteria ) {
      picked_move_x = mx;
      picked_move_y = my;
      best_criteria = criteria;
    }

    /* printf( "for move %zu, criteria is %f. Node has %llu wins and %llu games\n", move, criteria, move_node->win_cnt, move_node->game_cnt ); */
  }

  /* printf("picked: (%llu,%llu)\n", picked_move_x, picked_move_y); */

  assert(picked_move_x != (uint64_t)-1);
  assert(picked_move_y != (uint64_t)-1);

  *out_mx = picked_move_x;
  *out_my = picked_move_y;
}

static inline uint64_t
pick_next_move( game_table_t * gt,
                board_t        current_board,
                player_t       next_player )
{
  /* curr has less stones on it than any new nodes we may create.
     save it in case our slot in the table gets reused */

  node_t   curr    = *game_table_get( gt, current_board );
  uint64_t moves   = board_get_all_moves( &current_board, next_player );
  uint64_t n_moves = (uint64_t)__builtin_popcountll( moves );

  if( n_moves==0 ) return MOVE_PASS;

  // perform evaluation 100 times on each loop
  // some of what we pick will stick around
  for( size_t trial = 0; trial < 1000; ++trial ) {
    /* printf("-----TRIAL %zu\n", trial); */
    /* print_tree( gt, current_board, next_player, 0 ); */

    uint64_t mx, my;
    select_from_children( gt, curr, next_player, moves, n_moves, &mx, &my );

    /* printf("Selected immediate child of %d on (%llu,%llu)\n", next_player, mx, my); */

    board_t top = current_board;
    bool ret = board_make_move( &top, next_player, mx, my );
    assert( ret );

    // walk the move until we hit a child node that hasn't been expanded yet
    // or a terminal node.
    // max path length is 64 moves

    player_t winner;
    board_t  path[64]        = { top };
    size_t   n_moves_in_path = 1;
    player_t path_player     = !next_player;

    while( 1 ) {
      // grab top board from path and make a move
      top = path[n_moves_in_path-1];

      // if the game is over at this state, we are done
      if( board_is_game_over( &top, &winner ) ) break;

      // figure out what to do next
      node_t * path_node = game_table_get( gt, top );
      assert( path_node ); // table too small

      if( path_node->game_cnt > 0 ) {
        /* printf( "Found a previously visited node along path. Taking a new move\n" ); */
        // we've been here before, add a new child
        uint64_t path_moves   = board_get_all_moves( &top, path_player );
        uint64_t path_n_moves = (uint64_t)__builtin_popcountll( path_moves );

        if( path_n_moves==0 ) {
          // player has to pass, do nothing, loop will flip the player
        }
        else {
          uint64_t path_mx, path_my;
          select_from_children( gt, *path_node, path_player, path_moves, path_n_moves, &path_mx, &path_my );

          bool ret = board_make_move( &top, path_player, path_mx, path_my );
          assert( ret );
        }
      }
      else {
        // we've never been here. Add a randomized playout
        winner = play_randomly( top, path_player, current_board.white * current_board.black * trial );
        break;
      }

      /* printf("After applying path_move at idx %zu:\n", n_moves_in_path); */
      /* board_print( &top ); */

      // save our path and update the next player
      path[n_moves_in_path++] = top;
      path_player = !path_player;
    }

    // update the entire path with the winner

    for( size_t i = 0; i < n_moves_in_path; ++i ) {
      node_t * path_node = game_table_get( gt, path[i] );
      assert( path_node );

      path_node->win_cnt  += winner==next_player;
      path_node->game_cnt += 1;
    }
  }

  // now pick the move that maximizes likelyhood we win
  size_t move_idx      = (size_t)-1;
  float best_criteria = -1;

  for( size_t move = 0; move < n_moves; ++move ) {
    uint64_t mx, my;
    extract_move( moves, move, &mx, &my );

    board_t upd_board = curr.board;
    bool ret = board_make_move( &upd_board, next_player, mx, my );
    assert( ret );

    node_t * move_node = game_table_get( gt, upd_board );
    assert( move_node ); // table isn't big enough

    float criteria = 0.0f;
    if( move_node->game_cnt ) {
      criteria = (float)move_node->win_cnt / (float)move_node->game_cnt;
    }

    if( criteria > best_criteria ) {
      move_idx = move;
      best_criteria = criteria;
    }
  }

  uint64_t mx, my;
  extract_move( moves, move_idx, &mx, &my );
  return BIT_MASK( mx, my );
}
