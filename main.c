#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "pd_api.h"

#include "bitboard.h"

static PlaydateAPI* G_pd = NULL;

static int
board_newobject( lua_State * l )
{
  (void)l;

  board_t * board = G_pd->system->realloc( NULL, sizeof(board_t) );
  board_init( board );

  G_pd->lua->pushObject( board, "board", 0 );
  return 1;
}

static int
board_gc( lua_State * l )
{
  (void)l;

  board_t * board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !board ) return 1;

  G_pd->system->realloc( board, 0 );
  return 0;
}

int
board_get_cell( lua_State * l )
{
  (void)l;

  board_t * board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !board ) return 1;

  // FIXME should be checking if negative
  // but I just will never pass that
  // also note that lua one indexes, so I kept that convention in lua code

  uint64_t x = (uint64_t)G_pd->lua->getArgInt( 2 ) - 1;
  uint64_t y = (uint64_t)G_pd->lua->getArgInt( 3 ) - 1;

  // check if white or black
  uint64_t mask = BIT_MASK( x, y );

  bool has_white = board->white & mask;
  bool has_black = board->black & mask;

  if( has_white ) {
    if( has_black ) {
      G_pd->lua->pushInt( 3 ); // invalid
    }
    else {
      G_pd->lua->pushInt( 0 ); // white
    }
    return 1;
  }

  if( has_black ) {
    if( has_white ) {
      G_pd->lua->pushInt( 3 ); // invalid
    }
    else {
      G_pd->lua->pushInt( 1 );
    }
    return 1;
  }

  // else return nil?
  return 0;
}

int
l_board_make_move( lua_State * l )
{
  (void)l;

  board_t * board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !board ) return 1;

  // FIXME should be checking if negative
  // but I just will never pass that
  // 1 indexing again

  uint64_t x     = (uint64_t)G_pd->lua->getArgInt( 2 ) - 1;
  uint64_t y     = (uint64_t)G_pd->lua->getArgInt( 3 ) - 1;
  uint64_t color = (uint64_t)G_pd->lua->getArgInt( 4 );

  if( board_make_move( board, color, x, y ) ) {
    G_pd->lua->pushInt( 1 ); // return "true"
    return 1;
  }

  // return nil / false
  return 0;
}

int
l_board_can_move( lua_State * l )
{
  (void)l;

  board_t * board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !board ) return 0;

  uint64_t color = (uint64_t)G_pd->lua->getArgInt( 2 );

  if( 0 == board_get_all_moves( board, color ) ) {
    G_pd->lua->pushInt( 0 ); // return "true"
    return 1;
  }
  else {
    G_pd->lua->pushInt( 1 ); // return "true"
    return 1;
  }
}

static const lua_reg boardLib[] = {
  { "new",        board_newobject },
  { "__gc",       board_gc },
  { "get_cell",   board_get_cell },
  { "make_move",  l_board_make_move },
  { "can_move",   l_board_can_move },
};

int
eventHandler(PlaydateAPI* playdate, PDSystemEvent event, uint32_t arg)
{
  if( event == kEventInitLua ) {
    G_pd = playdate;

    char const * err;
    if( !G_pd->lua->registerClass( "board", boardLib, NULL, 0, &err ) ) {
      G_pd->system->logToConsole( "%s:%i: registerClass failed, %s", __FILE__, __LINE__, err );
    }
  }

  return 0;
}
