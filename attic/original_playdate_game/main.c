//#include <stdio.h>
//#include <stdlib.h>
#include <stdbool.h>

#include "pd_api.h"

#include "bitboard.h"
#include "game_tree.h"

typedef struct {
  board_t      board[1];
  game_table_t table[];         // trailing array member, kinda a hack
} l_board_t;

static PlaydateAPI * G_pd = NULL;

static int
l_board_newobject( lua_State * l )
{
  (void)l;

  _Static_assert(offsetof(l_board_t, table) == sizeof(board_t), "padding breaks alloc size");
  size_t sz = sizeof(board_t) + game_table_size( 8192 );

  l_board_t * l_board = G_pd->system->realloc( NULL, sz );
  board_init( l_board->board );

  game_table_new( l_board->table, 8192 ); // FIXME rename to init

  G_pd->lua->pushObject( l_board, "board", 0 );
  return 1;
}

static int
l_board_gc( lua_State * l )
{
  (void)l;

  board_t * board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !board ) return 1;

  G_pd->system->realloc( board, 0 );
  return 0;
}

static int
l_board_get_cell( lua_State * l )
{
  (void)l;

  l_board_t const * l_board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !l_board ) return 1;

  board_t const * board = l_board->board;

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

static int
l_board_make_move( lua_State * l )
{
  (void)l;

  l_board_t * l_board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !l_board ) return 1;

  board_t * board = l_board->board;

  // FIXME should be checking if negative
  // but I just will never pass that
  // 1 indexing again

  uint64_t x     = (uint64_t)G_pd->lua->getArgInt( 2 ) - 1;
  uint64_t y     = (uint64_t)G_pd->lua->getArgInt( 3 ) - 1;
  player_t color = (player_t)G_pd->lua->getArgInt( 4 );

  if( board_make_move( board, color, x, y ) ) {
    G_pd->lua->pushInt( 1 ); // return "true"
    return 1;
  }

  // return nil / false
  return 0;
}

static int
l_board_can_move( lua_State * l )
{
  (void)l;

  l_board_t const * l_board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !l_board ) return 0;

  board_t const * board = l_board->board;

  player_t color = (player_t)G_pd->lua->getArgInt( 2 );

  if( 0 == board_get_all_moves( board, color ) ) {
    // FIXME
    G_pd->lua->pushInt( 0 ); // return "false" (could also return nil)
    return 1;
  }
  else {
    G_pd->lua->pushInt( 1 ); // return "true"
    return 1;
  }
}

static int
l_board_game_over( lua_State * l )
{
  (void)l;

  l_board_t const * l_board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !l_board ) return 0;

  board_t const * board = l_board->board;

  player_t winner;
  if( board_is_game_over( board, &winner ) ) {
    G_pd->lua->pushInt( (int)winner ); // return the winner
    return 1;
  }
  else {
    // return nil
    return 0;
  }
  return 0;
}

static int
l_board_computer_take_turn( lua_State * l )
{
  (void)l;

  l_board_t * l_board = G_pd->lua->getArgObject( 1, "board", NULL );
  if( !l_board ) return 0;

  board_t *      board = l_board->board;
  game_table_t * table = l_board->table;

  // computer is white

  G_pd->system->logToConsole( "%s:%i: picking move", __FILE__, __LINE__ );
  uint64_t move = pick_next_move( table, *board, PLAYER_WHITE );
  G_pd->system->logToConsole( "%s:%i: move picked", __FILE__, __LINE__ );

  if( move != MOVE_PASS ) {
    // FIXME gross
    for( size_t x = 0; x < 8; ++x ) {
      for( size_t y = 0; y < 8; ++y ) {
        if( move & BIT_MASK(x,y) ) {
          bool ret = board_make_move( board, PLAYER_WHITE, x, y );
          return 0;
          (void)ret;
          //assert(ret);
        }
      }
    }
  }

  return 0;
}

static const lua_reg boardLib[] = {
  { "new",                l_board_newobject },
  { "__gc",               l_board_gc },
  { "get_cell",           l_board_get_cell },
  { "make_move",          l_board_make_move },
  { "can_move",           l_board_can_move },
  { "game_over",          l_board_game_over },
  { "computer_take_turn", l_board_computer_take_turn },
  { NULL, NULL }
};

int
eventHandler( PlaydateAPI*  playdate,
              PDSystemEvent event,
              uint32_t      arg )
{
  (void)arg;

  if( event == kEventInitLua ) {
    G_pd = playdate;

    char const * err;
    if( !G_pd->lua->registerClass( "board", boardLib, NULL, 0, &err ) ) {
      G_pd->system->logToConsole( "%s:%i: registerClass failed, %s", __FILE__, __LINE__, err );
    }

    G_pd->system->logToConsole( "%s:%i: API fully configured", __FILE__, __LINE__ );
  }

  return 0;
}
