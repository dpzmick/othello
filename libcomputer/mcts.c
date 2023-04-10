#include "mcts.h"
#include <stdio.h>

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

  game_tree_init( ret->tree, n_nodes );
}

static othello_game_t
select_best_child( game_tree_t *              tree,
                   uint64_t                   min_stones,
                   othello_game_t const *     parent,
                   othello_move_ctx_t const * parent_ctx,
                   uint64_t                   parent_game_cnt )
{
  uint64_t       own_moves     = parent_ctx->own_moves;
  uint64_t       n_own_moves   = parent_ctx->n_own_moves;
  othello_game_t best_child    = *parent;
  float          best_criteria = -1.0f; // always positive

  /* always valid to make. We might not have any moves. */
  othello_game_make_move( &best_child, parent_ctx, OTHELLO_MOVE_PASS );

  for( uint64_t move_idx = 0; move_idx < n_own_moves; ++move_idx ) {
    othello_game_t child;
    uint64_t       move;
    bool           valid;

    child = *parent;
    move  = keep_ith_set_bit( own_moves, move_idx );
    valid = othello_game_make_move( &child, parent_ctx, move );
    if( UNLIKELY( !valid ) ) Fail( "move is not valid" );

    /* Lookup this child node in the map. Note that we do not allow insertion
       here. If the node doesn't exist, we'll push it to stack and exit the
       loop on the next iteration. We could short circuit this to save the
       double lookup, but leaving it around for clarity and a single exit
       point. */

    game_tree_node_t * child_node = game_tree_get( tree, &child, min_stones, false );
    if( !child_node ) {
      best_child = child;
      break;
    }

    if( UNLIKELY( child_node->game_cnt==0 ) ) {
      Fail( "unexpanded child should have failed lookup" );
    }

    float win_cnt  = (float)child_node->win_cnt;
    float game_cnt = (float)child_node->game_cnt;

    float child_criteria = win_cnt/game_cnt
      + sqrtf( 2.0f ) * sqrtf( log2f( (float)parent_game_cnt ) / game_cnt );

    if( child_criteria > best_criteria ) {
      best_child    = child;
      best_criteria = child_criteria;
    }
  }

  return best_child;
}

static void
run_trial( mcts_state_t *         mcts,
           othello_game_t const * game,
           uint64_t               seed )
{
  /* Search the tree until we find an unexplored node. */

  game_tree_t * tree        = mcts->tree;
  uint64_t      min_stones  = othello_game_popcount( game );

  game_tree_node_t * unexplored_node = NULL;
  othello_game_t     path[64]        = { *game };
  size_t             n_path          = 1;

  while( !unexplored_node ) {
    if( UNLIKELY( n_path >= ARRAY_SIZE(path) ) ) Fail( "overflowed the path" );

    othello_game_t     stk_game[1] = { path[n_path-1] };
    uint8_t            stk_winner;
    othello_move_ctx_t stk_game_ctx[1];

    game_tree_node_t * stk_node = game_tree_get( tree, stk_game, min_stones, true );
    if( !stk_node ) Fail( "out of space in hash table" );

    if( stk_node->game_cnt == 0 ) {
      /* we've never been here */
      unexplored_node = stk_node;
      break;
    }

    /* If this node ends the game, we've fully explored some path. Return this
       node as our unexplored node so that we'll "playout" from here (figure out
       the winner), and update the paths. Unlikely because most games are not
       over. */

    if( UNLIKELY( !othello_game_start_move( stk_game, stk_game_ctx, &stk_winner ) ) ) {
      unexplored_node = stk_node;
      break;
    }

    path[n_path++] = select_best_child( tree, min_stones,
                                        stk_game, stk_game_ctx, stk_node->game_cnt );
  }

  /* We have unexplored node and the path we took to get to it */

  othello_game_t _tmp   = *unexplored_node->game;
  uint8_t        winner = othello_game_random_playout( &_tmp, seed );

  /* Update everything in the path. All should be in table because we just
     looked them up. */

  for( size_t i = 0; i < n_path; ++i ) {
    game_tree_node_t * path_node = game_tree_get( tree, &path[i], min_stones, false );
    if( UNLIKELY( !path_node ) ) Fail( "impossible" );

    path_node->win_cnt  += winner==mcts->play_as;
    path_node->game_cnt += 1;
  }
}

#if 0
static void
dump_tree( game_tree_t const * tree,
           uint64_t            min_stones )
{
  for( size_t i = 0; i < tree->n_nodes; ++i ) {
    game_tree_node_t const * node             = &tree->nodes[i];
    othello_game_t const *   node_game        = node->game;
    size_t                   node_game_popcnt = othello_game_popcount( node_game );

    if( node_game_popcnt < min_stones ) continue;
    if( node->win_cnt == 0 ) continue;

    printf( "----------------------\n" );
    othello_board_print( node_game );
    printf( "wins: %llu, cnt: %llu\n", node->win_cnt, node->game_cnt );
  }
}
#endif

uint64_t
mcts_select_move( mcts_state_t *             mcts,
                  othello_game_t const *     game,
                  othello_move_ctx_t const * ctx )
{
  /* assumes that the game is some game from later in the game than boards we've
     already seen */

  game_tree_t * tree       = mcts->tree;
  size_t        trials     = mcts->trials;
  uint64_t      moves      = ctx->own_moves;
  uint64_t      n_moves    = ctx->n_own_moves;
  uint64_t      min_stones = othello_game_popcount( game );

  if( UNLIKELY( game->curr_player!=mcts->play_as ) ) Fail( "Computer only plays as %d", mcts->play_as );
  if( n_moves==0 ) return OTHELLO_MOVE_PASS;

  /* Run enough trials to ensure we expand all immediate children */
  uint64_t now = wallclock();
  for( size_t trial = 0; trial < MAX( trials, n_moves ); ++trial ) {
    run_trial( mcts, game, hash_u64( trial + now ) );
  }

  uint64_t best_move     = OTHELLO_MOVE_PASS;
  float    best_criteria = -1;

  for( uint64_t move_idx = 0; move_idx < n_moves; ++move_idx ) {
    /* copy the game state */
    othello_game_t updated_game[1] = { *game };
    uint64_t       move            = keep_ith_set_bit( moves, move_idx );

    /* apply the new move to the game */
    bool valid = othello_game_make_move( updated_game, ctx, move );
    if( UNLIKELY( !valid ) ) Fail( "attempted to apply invalid move" );

    /* We might not have expanded all of our children */
    game_tree_node_t * move_node = game_tree_get( tree, updated_game, min_stones, false );
    if( UNLIKELY( !move_node ) ) {
      /* printf( "failed at: %llu %llu\n", updated_game->white, updated_game->black ); */
      /* othello_board_print( game ); */
      /* othello_board_print( updated_game ); */
      Fail( "didn't expand all children" );
    }

    float criteria = (float)move_node->win_cnt / (float)move_node->game_cnt;

    if( criteria > best_criteria ) {
      best_move = move;
      best_criteria = criteria;
    }
  }

  /* dump_tree( tree, min_stones ); */

  return best_move;
}
