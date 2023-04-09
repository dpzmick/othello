#include "mcts.h"

#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libothello/othello.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// Game trees stored in a hash table
//
// looking up the current game state (if we've seen it before) is easy
//
// we can invalidate slots implicitly by checking if their state is still
// reachable. If it's not, we can reuse the slot
//
// "traversal" is modifying the game state and looking up the next node

typedef struct {
  othello_game_t game[1];       // game state stored in this node
  uint64_t       win_cnt;       // number of simulated wins from this position
  uint64_t       game_cnt;      // number of simulated games played from this node
} node_t;

// 16 KiB of D-cache means we can fit 512 entries of this table into cache
// whole ram is 16 MiB

struct mcts_state {
  size_t trials;
  size_t n_nodes;
  size_t mask;
  size_t n_gets;                // stats
  size_t n_loops;               // stats
  node_t nodes[];
};

// We can't really estimate the load factor and rehash "sometimes" since cells
// become tombstoned automatically.
//
// This structure needs to be used for a long time so we will need to deal with
// tombstones increasing the load factor too much... so, we'll do cleanup as we
// go

size_t
mcts_state_size( size_t n_nodes )
{
  n_nodes = next_pow2( n_nodes );
  return sizeof(mcts_state_t) + n_nodes*sizeof(node_t);
}

mcts_state_t *
mcts_state_init( void * mem,
                 size_t trials,
                 size_t n_nodes )
{
  mcts_state_t * ret = mem;
  ret->trials  = trials;
  ret->n_nodes = next_pow2( n_nodes );
  ret->mask    = ret->n_nodes - 1;
  ret->n_gets  = 0;
  ret->n_loops = 0;

  memset( ret->nodes, 0, sizeof(node_t)*ret->n_nodes );

  return ret;
}

// node is valid until the next call to get

static inline node_t *
mcts_get( mcts_state_t *         mcts,
          othello_game_t const * game,
          uint64_t               min_stones,
          bool                   allow_insert )
{
  uint64_t hash    = othello_game_hash( game );
  size_t   mask    = mcts->mask;
  node_t * nodes   = mcts->nodes;
  size_t   slot    = hash & mask;
  node_t * ret     = NULL;
  size_t   n_loops = 0;

  while( 1 ) {
    node_t *         node             = &nodes[slot];
    othello_game_t * node_game        = node->game;
    size_t           node_game_popcnt = othello_game_popcount( node_game );

    n_loops += 1;

    // found the matching cell
    if( LIKELY( othello_game_eq( game, node_game ) ) ) {
      ret = node;
      goto done;
    }

    // boards always evolve by adding new stones
    // a cell is "empty" if it contains an earlier game state that we no longer care about
    if( allow_insert && min_stones > node_game_popcnt ) {
      // reset node and use it
      *node->game    = *game;
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
  mcts->n_gets += 1;
  mcts->n_loops += n_loops;
  return ret;
}

/* Select a move to take given the current state of the game tree.
   Used _internally_. The external selection runs many trials */

static uint64_t
ucb_select_move( mcts_state_t *         mcts,
                 othello_game_t const * game,
                 node_t const *         game_node,
                 uint64_t               valid_moves,
                 size_t                 n_moves,
                 uint64_t               min_stones )
{
  uint64_t best_move     = 0;
  float    best_criteria = -1.0f; // all computed criteria are positive

  for( size_t move_idx = 0; move_idx < n_moves; ++move_idx ) {
    /* select the ith valid move */
    uint64_t move = keep_ith_set_bit( valid_moves, move_idx );

    /* copy the game state */
    othello_game_t updated_game[1] = { *game };

    /* apply the new move to the game */
    bool valid = othello_game_make_move( updated_game, move );
    if( UNLIKELY( !valid ) ) Fail( "attempted to apply invalid move" );

    /* Lookup the state of the game. */

    node_t * move_node = mcts_get( mcts, updated_game, min_stones, true );
    if( !move_node ) Fail( "Out of space in the hash table" );

    /* If we've never been there before, we're done; we will run a randomized
       playout in the outer function. We value exploration. */

    if( move_node->game_cnt==0 ) {
      return move;
    }

    float criteria = 0.0f;
    if( move_node->game_cnt > 0 ) {
      criteria += (float)move_node->win_cnt / (float)move_node->game_cnt;
    }

    if( game_node->game_cnt > 0 && move_node->game_cnt > 0 ) {
      criteria += sqrtf(2.0) * sqrtf(log2f((float)game_node->game_cnt) / (float)move_node->game_cnt);
    }

    if( criteria > best_criteria ) {
      best_move = move;
      best_criteria = criteria;
    }
  }

  return best_move;
}

uint64_t
mcts_select_move( mcts_state_t *         mcts,
                  othello_game_t const * game )
{
  /* assumes that the game is some game from later in the game than boards we've
     already seen */

  if( game->curr_player!=OTHELLO_BIT_WHITE ) Fail( "Computer only plays as white" );

  size_t   trials     = mcts->trials;
  uint64_t min_stones = othello_game_popcount( game );
  uint64_t moves      = othello_game_all_valid_moves( game );
  size_t   n_moves    = (size_t)__builtin_popcountll( moves );
  if( n_moves==0 ) return OTHELLO_MOVE_PASS;

  node_t * curr = mcts_get( mcts, game, min_stones, true );
  if( !curr ) Fail( "Out of space in hash table" );

  for( size_t trial = 0; trial < trials; ++trial ) {
    uint64_t move = ucb_select_move( mcts, game, curr, moves, n_moves, min_stones );

    othello_game_t top = *game;
    othello_game_make_move( &top, move );

    /* Walk the path from this moved until we hit a child node that hasn't been
       expanded yet. Note that max path length is 64 moves. */

    uint8_t        winner;
    othello_game_t path[64]        = { top };
    size_t         n_moves_in_path = 1;

    while( 1 ) {
      top = path[n_moves_in_path-1];
      if( othello_game_is_over( &top, &winner ) ) break;

      node_t * path_node = mcts_get( mcts, &top, min_stones, true );
      if( !path_node ) Fail( "Out of space in hash table" );

      if( path_node->game_cnt > 0 ) {
        /* we've been here before, add a new child */
        uint64_t path_moves   = othello_game_all_valid_moves( &top );
        uint64_t path_n_moves = (uint64_t)__builtin_popcountll( path_moves );

        if( path_n_moves==0 ) {
          othello_game_make_move( &top, OTHELLO_MOVE_PASS );
        }
        else {
          uint64_t move = ucb_select_move( mcts, &top, path_node, path_moves, path_n_moves, min_stones );
          othello_game_make_move( &top, move );
        }
      }
      else {
        /* we've never been here. Add a randomized playout */
        uint64_t seed = game->white * game->black * trial;
        winner = othello_game_random_playout( &top, seed ); // modifies top
        break; // !important
      }

      path[n_moves_in_path++] = top;
    }

    /* /update the entire path with the winner */

    for( size_t i = 0; i < n_moves_in_path; ++i ) {
      node_t * path_node = mcts_get( mcts, &path[i], min_stones, false );
      if( !path_node ) Fail( "impossible" );

      path_node->win_cnt  += winner==OTHELLO_BIT_WHITE;
      path_node->game_cnt += 1;
    }
  }

  // now pick the move that maximizes likelyhood we win
  uint64_t best_move     = OTHELLO_MOVE_PASS;
  float    best_criteria = -1;

  for( size_t move_idx = 0; move_idx < n_moves; ++move_idx ) {
    uint64_t move = keep_ith_set_bit( moves, move_idx );

    /* copy the game state */
    othello_game_t updated_game[1] = { *game };

    /* apply the new move to the game */
    bool valid = othello_game_make_move( updated_game, move );
    if( UNLIKELY( !valid ) ) Fail( "attempted to apply invalid move" );

    node_t * move_node = mcts_get( mcts, updated_game, min_stones, false );
    if( !move_node ) Fail( "Impossible" );

    float criteria = 0.0f;
    if( move_node->game_cnt ) {
      criteria = (float)move_node->win_cnt / (float)move_node->game_cnt;
    }

    if( criteria > best_criteria ) {
      best_move = move;
      best_criteria = criteria;
    }
  }

  return best_move;
}
