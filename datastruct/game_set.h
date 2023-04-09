#pragma once

#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libothello/othello.h"

#include <string.h>

/* Hash set of games, not really very clever.

   Deletion not allowed as it is a pain to implement and who really needs
   deletion anyway. */

typedef struct {
  size_t         n_slots;
  size_t         mask;
  size_t         n_gets;        // stats
  size_t         n_loops;       // stats
  othello_game_t slots[];
} game_set_t;

static inline size_t
game_set_size( size_t n_nodes )
{
  n_nodes = next_pow2( n_nodes );
  return sizeof(game_set_t) + n_nodes*sizeof(othello_game_t);
}

static inline game_set_t *
game_set_new( void * mem,
              size_t n_games )
{
  game_set_t * ret = mem;
  ret->n_slots = next_pow2( n_games );
  ret->mask    = ret->n_slots - 1;
  ret->n_gets  = 0;
  ret->n_loops = 0;

  /* Being a bit sneaky here with the state of the games.

     Prefault memory by writing a non-zero sentinel to all of them, but
     otherwise leave totally uninitialized. This is also getting a bit into the
     inner guts of the othello game struct, but this is a small program so who
     cares about super great design? */

  for( size_t i = 0; i < ret->n_slots; ++i ) {
    ret->slots[i].curr_player = OTHELLO_GAME_SET_SENTINEL;
  }

  return ret;
}

/* No deletion, so this pointer is valid forever. Don't modify it */

static inline othello_game_t const *
game_set_get( game_set_t *           s,
              othello_game_t const * game,
              bool                   allow_create )
{
  size_t                 mask    = s->mask;
  othello_game_t *       slots   = s->slots;
  uint64_t               hash    = othello_game_hash( game );
  size_t                 slot    = hash & mask;
  othello_game_t const * ret     = NULL;
  size_t                 n_loops = 0;

  while( 1 ) {
    othello_game_t * slot_game = &slots[slot];

    n_loops += 1;

    /* Do we already have this one?

       NOTE: this will be an uninitialized read if the slot hasn't been written
       to yet, but game_eq is guaranteed to fail b.c. we initialized the player
       to some bogus value that only we use. Valgrind will whine, but it doesn't
       work on macOS anyway ;P */

    if( LIKELY( othello_game_eq( game, slot_game ) ) ) {
      ret = slot_game;
      goto done;
    }

    if( allow_create && slot_game->curr_player == OTHELLO_GAME_SET_SENTINEL ) {
      memcpy( slot_game, game, sizeof(*game) );
      ret = slot_game;
      goto done;
    }

    slot = (slot+1) & mask;

    if( UNLIKELY( slot == (hash & mask) ) ) { // out of space, we wrapped around
      ret = NULL;
      goto done;
    }
  }

done:
  s->n_gets += 1;
  s->n_loops += n_loops;
  return ret;
}

/* _Compute_ the number of slots occupied in the table. Done as a calculation to
    keep struct all nice and aligned and b.c. we only call this once. */

static inline size_t
game_set_n_occupied( game_set_t const * s )
{
  othello_game_t const * slots   = s->slots;
  size_t                 n_slots = s->n_slots;

  size_t cnt = 0;
  for( size_t i = 0; i < n_slots; ++i ) {
    othello_game_t const * slot_game = &slots[i];
    cnt += slot_game->curr_player != OTHELLO_GAME_SET_SENTINEL;
  }

  return cnt;
}
