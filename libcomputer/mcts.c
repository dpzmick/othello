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
  size_t n_nodes;
  size_t mask;
  size_t n_gets;                // stats
  size_t n_loops;               // stats
  size_t n_null_rets;           // stats
  game_tree_node_t __attribute__ ((aligned (32))) nodes[];
} game_tree_t;

static size_t
game_tree_size( size_t n_nodes )
{
  n_nodes = (size_t)next_pow2( n_nodes );
  return sizeof(game_tree_t) + n_nodes*sizeof(game_tree_node_t);
}

static void
game_tree_init( game_tree_t * tree,
                size_t        n_nodes )
{
  tree->n_nodes     = (size_t)next_pow2( n_nodes );
  tree->mask        = tree->n_nodes - 1;
  tree->n_gets      = 0;
  tree->n_loops     = 0;
  tree->n_null_rets = 0;

  memset( tree->nodes, 0, sizeof(game_tree_node_t)*tree->n_nodes );

  /* Mark every slot as "never used" via the sentinel curr_player. Probes
     terminate at a sentinel slot (no real game can have curr_player ==
     SENTINEL, so it never matches), giving lookup an O(chain) bound
     instead of O(table). */
  for( size_t i = 0; i < tree->n_nodes; i++ ) {
    tree->nodes[i].game->curr_player = OTHELLO_GAME_SET_SENTINEL;
  }
}

/* node is valid until the next call to get */

/* Open-addressed lookup with implicit GC: a slot is "free" iff its stored
   game has popcount < min_stones, since Othello stones only ever accumulate.
   A slot whose curr_player == SENTINEL has never been written and acts as
   the chain terminator (no real game has curr_player == SENTINEL, so it
   never matches).

   Subtlety: an earlier version claimed the first stale slot it saw during
   an insert and returned. That created duplicate entries -- suppose game
   G was previously inserted at slot S+5 (because S..S+4 were valid when G
   was first inserted). Later, min_stones rises and slot S+2 becomes stale.
   A new lookup of G with allow_insert=true would walk S, S+1, hit S+2 as
   stale, claim it, and never reach the real entry at S+5. G then exists
   in two slots; the older entry's stats get orphaned until min_stones
   rises enough to evict that one too. Effect on play strength is small
   (MCTS "forgets" some accumulated stats) but it's a real correctness
   issue and was probably costing some of the noise in thunderdome runs.

   Fix: remember the first stale slot we see, keep probing for a real
   match, and only claim the remembered slot when we hit the chain
   terminator (sentinel) or wrap. Costs ~1 extra probe per insert.

   This whole scheme is a domain-specific lazy-deletion in an open-address
   table -- it works because Othello popcount is monotonic, but the
   interleaving of stale and valid slots that gradual min_stones growth
   produces is exactly the canonical lazy-deletion failure mode. A
   bump-allocated explicit tree (reset per search) would sidestep all of
   this and be substantially faster; left as future work for when we
   replace random-rollout MCTS with NN-guided search. */

static game_tree_node_t *
//__attribute__((noinline))
game_tree_get( game_tree_t *          tree,
               othello_game_t const * game,
               uint64_t               min_stones,
               bool                   allow_insert )
{
  uint64_t           hash        = othello_game_hash( game );
  size_t             mask        = tree->mask;
  game_tree_node_t * nodes       = tree->nodes;
  size_t             start_slot  = hash & mask;
  size_t             slot        = start_slot;
  game_tree_node_t * ret         = NULL;
  game_tree_node_t * first_stale = NULL; // earliest reusable slot in the chain
  size_t             n_loops     = 0;

#if 0
  if( min_stones < tree->max_min_stones ) Fail( "min stones cannot go backwards" );
  tree->max_min_stones = MAX( tree->max_min_stones, min_stones );
#endif

  while( 1 ) {
    game_tree_node_t * node      = &nodes[slot];
    othello_game_t *   node_game = node->game;

    n_loops += 1;

    bool is_empty = ( node_game->curr_player == OTHELLO_GAME_SET_SENTINEL );

    // found the matching cell. (Sentinel slots can't match -- their curr_player
    // doesn't correspond to any real player.)
    if( LIKELY( !is_empty && othello_game_eq( game, node_game ) ) ) {
      ret = node;
      goto done;
    }

    // empty (sentinel) terminates the chain: a match cannot exist beyond here
    if( is_empty ) {
      if( allow_insert ) {
        // claim the earliest stale slot we found, or this empty slot if none
        game_tree_node_t * dst = first_stale ? first_stale : node;
        *dst->game    = *game;
        dst->win_cnt  = 0;
        dst->game_cnt = 0;
        ret = dst;
      } else {
        ret = NULL;
      }
      goto done;
    }

    // remember (but don't claim yet) the first stale slot in this chain --
    // a real match might still exist further on
    if( allow_insert && !first_stale &&
        othello_game_popcount( node_game ) < min_stones ) {
      first_stale = node;
    }

    slot = (slot+1) & mask;

    if( UNLIKELY( slot == start_slot ) ) {
      // walked the whole table without finding a match or a sentinel.
      // Use the stale slot if we saw one; otherwise truly out of space.
      if( allow_insert && first_stale ) {
        *first_stale->game    = *game;
        first_stale->win_cnt  = 0;
        first_stale->game_cnt = 0;
        ret = first_stale;
      } else {
        tree->n_null_rets += 1;
        ret = NULL;
      }
      goto done;
    }
  }

done:
  tree->n_gets += 1;
  tree->n_loops += n_loops;
  return ret;
}

// -------------------

#include <inttypes.h>

static void
dump_json( FILE *                 f,
           game_tree_t *          tree,
           uint64_t               min_stones,
           othello_game_t const * root,
           uint64_t               parent_game_cnt,
           bool                   first )
{
  game_tree_node_t const * node = game_tree_get( tree, root, min_stones, false );
  if( !node ) return;

  if( !first ) fprintf( f, ", " );
  fprintf( f, "\n{" );
  fprintf( f, "\"wins\": %" PRIu64, node->win_cnt );
  fprintf( f, ",\"games\": %" PRIu64, node->game_cnt );
  if( parent_game_cnt ){
    float win_cnt  = (float)node->win_cnt;
    float game_cnt = (float)node->game_cnt;

    float criteria = win_cnt/game_cnt
      + sqrtf( 2.0f ) * sqrtf( log2f( (float)parent_game_cnt ) / game_cnt );
    fprintf( f, ",\"criteria\": %0.3f", (double)criteria );
  }
  fprintf( f, ",\"board\": [" );

  // FIXME add UCB score to children

  uint64_t moves = othello_game_all_valid_moves( root );
  for( size_t y = 0; y < 8; ++y ) {

    if( y!=0 ) fprintf( f, ", " );
    fprintf( f, "[" );
    for( size_t x = 0; x < 8; ++x ) {
      if( x!=0 ) fprintf( f, ", " );

      uint64_t mask = othello_bit_mask(x,y);
      uint8_t bit_white = (root->white & mask) != 0;
      uint8_t bit_black = (root->black & mask) != 0;
      bool    is_move   = moves&mask;

      if( bit_white && bit_black ) {
        fprintf( f, "\"X\"" ); // invalid
      }
      else if( bit_white ) {
        fprintf( f, "\"W\"" );
      }
      else if( bit_black ) {
        fprintf( f, "\"B\"" );
      }
      else {
        if( is_move ) {
          fprintf( f, "\".\"" );
        }
        else {
          fprintf( f, "\" \"" );
        }
      }
    }
    fprintf( f, "]" );
  }
  fprintf( f, "]" );

  /* children defined by moves, check if we have any */

  uint8_t            winner;
  othello_move_ctx_t ctx[1];
  othello_game_start_move( root, ctx, &winner );

  fprintf( f, ", \"children\": [" );

  bool first_child = true;
  for( size_t i = 0; i < ctx->n_own_moves; ++i ) {
    othello_game_t upd   = *root;
    uint64_t       move  = keep_ith_set_bit( ctx->own_moves, i );
    bool           valid = othello_game_make_move( &upd, ctx, move );
    if( UNLIKELY( !valid ) ) Fail( "move is not valid" );

    game_tree_node_t const * node = game_tree_get( tree, &upd, min_stones, false );
    if( !node ) continue;

    // hopefully tree not too deep!
    dump_json( f, tree, min_stones, &upd, node->game_cnt, first_child );
    first_child = false;
  }

  fprintf( f, "]" );
  fprintf( f, "}" );
}

static void
dump_tree( game_tree_t *          tree,
           uint64_t               min_stones,
           othello_game_t const * root )
{
  /* char * buf; */
  /* size_t n; */
  /* FILE * f = open_memstream( &buf, &n ); */
  FILE * f = fopen( "data.js", "w" );
  fprintf( f, "var DATA = " );
  dump_json( f, tree, min_stones, root, 0, true );
  fclose( f );

  /* return buf; */
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

void
mcts_get_stats( mcts_state_t const * mcts,
                uint64_t *           out_loops,
                uint64_t *           out_gets )
{
  *out_gets  = mcts->tree->n_gets;;
  *out_loops = mcts->tree->n_loops;
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

    /* If we have never seen this child, select it */
    game_tree_node_t * child_node = game_tree_get( tree, &child, min_stones, true );
    if( !child_node ) Fail( "out of space in the hash table" );

    if( UNLIKELY( child_node->game_cnt==0 ) ) {
      /* could technically return the tree node here to save an extra loopup in
         the outer loop */
      return child;
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

    if( UNLIKELY( !othello_game_start_move( stk_game, stk_game_ctx, &stk_winner ) ) ) {
      unexplored_node = stk_node;
      break;
    }

    path[n_path++] = select_best_child( tree, min_stones,
                                        stk_game, stk_game_ctx, stk_node->game_cnt );
  }

  /* We have unexplored node and the path we took to get to it */

  uint64_t       _foo;
  othello_game_t _tmp = *unexplored_node->game;
  uint8_t        winner = othello_game_random_playout( &_tmp, seed, &_foo );

  /* Update everything in the path. All should be in table because we just
     looked them up. */

  for( size_t i = 0; i < n_path; ++i ) {
    game_tree_node_t * path_node = game_tree_get( tree, &path[i], min_stones, false );
    if( UNLIKELY( !path_node ) ) Fail( "impossible" );

    path_node->win_cnt  += winner==mcts->play_as;
    path_node->game_cnt += 1;
  }
}

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

  /* Run enough trials to ensure we expand all immediate children. Must be at
     least n_moves+1. +1 to explore _this node_ if never explored. */

  uint64_t now = wallclock();
  for( size_t trial = 0; trial < MAX( trials, n_moves+1 ); ++trial ) {
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

    /* Doing at least n_moves trials should ensure we always expand all children */
    game_tree_node_t * move_node = game_tree_get( tree, updated_game, min_stones, false );
    if( UNLIKELY( !move_node ) ) {
      dump_tree( tree, min_stones, game );
      Fail( "didn't expand all children" );
    }

    float criteria = (float)move_node->win_cnt / (float)move_node->game_cnt;

    if( criteria > best_criteria ) {
      best_move = move;
      best_criteria = criteria;
    }
  }


  return best_move;
}
