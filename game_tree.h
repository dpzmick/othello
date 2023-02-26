#pragma once

#include "bitboard.h"

#include <stdbool.h>
#include <stdint.h>
#include <xxhash.h>

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
  uint64_t wins;                // number of simulated wins from this position
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
game_table_get( game_table_t * gt,
                board_t        board )
{
  size_t   board_moves = board_total_moves( &board );
  size_t   mask        = gt->mask;
  node_t * nodes       = gt->nodes;
  uint64_t hash        = XXH3_64bits( &board, sizeof(board) );
  size_t   slot        = hash & mask;
  node_t * ret         = NULL;
  size_t   n_loops     = 0;

  while( 1 ) {
    node_t * node             = &nodes[slot];
    board_t  node_board       = node->board;
    size_t   node_board_moves = board_total_moves( &node_board );

    n_loops += 1;

    // found the matching cell
    if( LIKELY( board_eq( &node_board, &board ) ) ) {
      ret = node;
      goto done;
    }

    // boards always evolve by adding new moves
    // a cell is "empty" if it contains an earlier game state that we no longer care about
    if( board_moves > node_board_moves ) {
      // reset node and use it
      node->board    = board;
      node->wins     = 0;
      node->game_cnt = 0;
      ret = node;
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
