#include "mcts.h"

#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libothello/othello.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Game trees stored in a hash table for MCTS

   We can invalidate slots implicitly by checking if their state is still
   reachable. If it's not, we can reuse the slot. This does mean we can't really
   estimate the load factor and rehash "sometimes" since cells become tombstoned
   automatically, but we don't need to rehash.

   "traversal" is modifying the game state and looking up the next node

   16 KiB of D-cache means we can fit 512 entries of this table into cache
   whole ram is 16 MiB. */

typedef struct {
  othello_game_t game[1];       // game state stored in this node
  uint64_t       win_cnt;       // number of simulated wins from this position
  uint64_t       game_cnt;      // number of simulated games played from this node
} game_tree_node_t;

typedef struct {
  size_t           n_nodes;
  size_t           mask;
  size_t           n_gets;      // stats
  size_t           n_loops;     // stats
  game_tree_node_t nodes[];
} game_tree_t;

static inline size_t
game_tree_size( size_t n_nodes )
{
  n_nodes = (size_t)next_pow2( n_nodes );
  return sizeof(game_tree_t) + n_nodes*sizeof(game_tree_node_t);
}

static inline void
game_tree_init( game_tree_t * tree,
                size_t        n_nodes )
{
  tree->n_nodes = (size_t)next_pow2( n_nodes );
  tree->mask    = tree->n_nodes - 1;
  tree->n_gets  = 0;
  tree->n_loops = 0;

  memset( tree->nodes, 0, sizeof(game_tree_node_t)*tree->n_nodes );
}

/* node is valid until the next call to get */

static inline game_tree_node_t *
game_tree_get( game_tree_t *          tree,
               othello_game_t const * game,
               uint64_t               min_stones,
               bool                   allow_insert )
{
  uint64_t           hash    = othello_game_hash( game );
  size_t             mask    = tree->mask;
  game_tree_node_t * nodes   = tree->nodes;
  size_t             slot    = hash & mask;
  game_tree_node_t * ret     = NULL;
  size_t             n_loops = 0;

  while( 1 ) {
    game_tree_node_t * node             = &nodes[slot];
    othello_game_t *   node_game        = node->game;
    size_t             node_game_popcnt = othello_game_popcount( node_game );

    n_loops += 1;

    // found the matching cell
    if( LIKELY( othello_game_eq( game, node_game ) ) ) {
      ret = node;
      goto done;
    }

    // boards always evolve by adding new stones
    // a cell is "empty" if it contains an earlier game state that we no longer care about
    if( allow_insert && node_game_popcnt < min_stones ) {
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
  tree->n_gets += 1;
  tree->n_loops += n_loops;
  return ret;
}

// ---------------

struct mcts_state {
  size_t      trials;
  uint8_t     play_as;
  uint64_t    seed;
  game_tree_t tree[];
};

size_t
mcts_state_size( size_t n_nodes )
{
  return sizeof(mcts_state_t) + game_tree_size( n_nodes );
}

void
mcts_state_init( mcts_state_t * ret,
                 size_t         trials,
                 uint8_t        play_as,
                 uint64_t       seed,
                 size_t         n_nodes )
{
  ret->trials  = trials;
  ret->play_as = play_as;
  ret->seed    = seed;

  // FIXME assuming that trials >= number of possible moves

  game_tree_init( ret->tree, n_nodes );
}

/* Select a move to take given the current state of the game tree.
   Used _internally_. The external selection runs many trials */

static uint64_t
ucb_select_move( mcts_state_t *             mcts,
                 othello_game_t const *     game,
                 othello_move_ctx_t const * ctx,
                 game_tree_node_t const *   game_node,
                 uint64_t                   valid_moves,
                 uint64_t                   n_moves,
                 uint64_t                   min_stones )
{
  game_tree_t * tree          = mcts->tree;
  uint64_t      best_move     = 0;
  float         best_criteria = -1.0f; // all computed criteria are positive

  for( uint64_t move_idx = 0; move_idx < n_moves; ++move_idx ) {
    /* select the ith valid move */
    uint64_t move = keep_ith_set_bit( valid_moves, move_idx );

    /* copy the game state */
    othello_game_t updated_game[1] = { *game };

    /* apply the new move to the game */
    bool valid = othello_game_make_move( updated_game, ctx, move );
    if( UNLIKELY( !valid ) ) Fail( "attempted to apply invalid move" );

    /* Lookup the state of the game. */

    game_tree_node_t * move_node = game_tree_get( tree, updated_game, min_stones, true );
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
mcts_select_move( mcts_state_t *             mcts,
                  othello_game_t const *     game,
                  othello_move_ctx_t const * ctx )
{
  /* assumes that the game is some game from later in the game than boards we've
     already seen */

  if( game->curr_player!=mcts->play_as ) Fail( "Computer only plays as %d", mcts->play_as );

  game_tree_t *      tree       = mcts->tree;
  size_t             trials     = mcts->trials;
  uint64_t           min_stones = othello_game_popcount( game );

  uint64_t moves   = ctx->own_moves;
  uint64_t n_moves = ctx->n_own_moves;
  if( n_moves==0 ) return OTHELLO_MOVE_PASS;

  game_tree_node_t * curr = game_tree_get( tree, game, min_stones, true );
  if( !curr ) Fail( "Out of space in hash table" );

  for( size_t trial = 0; trial < trials; ++trial ) {
    uint64_t move = ucb_select_move( mcts, game, ctx, curr, moves, n_moves, min_stones );

    othello_game_t top = *game;
    othello_game_make_move( &top, ctx, move );

    /* Walk the path from this moved until we hit a child node that hasn't been
       expanded yet. Note that max path length is 64 moves. */

    uint8_t            winner;
    othello_move_ctx_t top_ctx[1];
    othello_game_t     path[64]        = { top };
    size_t             n_moves_in_path = 1;

    while( 1 ) {
      top = path[n_moves_in_path-1];
      if( !othello_game_start_move( &top, top_ctx, &winner )) {
        /* someone won */
        break;
      }

      if( n_moves_in_path >= 64 ) Fail( "too many moves" );

      game_tree_node_t * path_node = game_tree_get( tree, &top, min_stones, true );
      if( !path_node ) Fail( "Out of space in hash table" );

      if( path_node->game_cnt > 0 ) {
        /* we've been here before, add a new child */
        uint64_t path_moves   = top_ctx->own_moves;
        uint64_t path_n_moves = top_ctx->n_own_moves;

        if( path_n_moves==0 ) {
          bool valid = othello_game_make_move( &top, top_ctx, OTHELLO_MOVE_PASS );
          if( !valid ) Fail( "invalid move" );
        }
        else {
          uint64_t move = ucb_select_move( mcts, &top, top_ctx, path_node, path_moves, path_n_moves, min_stones );
          bool valid = othello_game_make_move( &top, top_ctx, move );
          if( !valid ) Fail( "invalid move" );
        }
      }
      else {
        /* we've never been here. Add a randomized playout */
        uint64_t seed = mcts->seed * game->white * game->black * trial;
        winner = othello_game_random_playout( &top, seed ); // modifies top
        break; // !important
      }

      path[n_moves_in_path++] = top;
    }

    /* update the entire path with the winner */

    for( size_t i = 0; i < n_moves_in_path; ++i ) {
      // FIXME make sure we actually for sure expand all children
      // we aren't right now
      game_tree_node_t * path_node = game_tree_get( tree, &path[i], min_stones, true );
      if( !path_node ) Fail( "impossible" );

      path_node->win_cnt  += winner==mcts->play_as;
      path_node->game_cnt += 1;
    }
  }

  // now pick the move that maximizes likelyhood we win
  uint64_t best_move     = OTHELLO_MOVE_PASS;
  float    best_criteria = -1;

  for( uint64_t move_idx = 0; move_idx < n_moves; ++move_idx ) {
    uint64_t move = keep_ith_set_bit( moves, move_idx );

    /* copy the game state */
    othello_game_t updated_game[1] = { *game };

    /* apply the new move to the game */
    bool valid = othello_game_make_move( updated_game, ctx, move );
    if( UNLIKELY( !valid ) ) Fail( "attempted to apply invalid move" );

    game_tree_node_t * move_node = game_tree_get( tree, updated_game, min_stones, false );
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
