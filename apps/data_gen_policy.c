#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libcomputer/nn.h"
#include "../libothello/othello.h"
#include "../misc/wthor.h"
#include "zstd_file.h"

#include <fcntl.h>
#include <inttypes.h>
#include <libgen.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <toml.h>
#include <unistd.h>

/// ----------------------------------
/// config

#define CONFIG_FILES_MAX 64

typedef struct {
  // inputs
  char   wthor_filenames[CONFIG_FILES_MAX][PATH_MAX+1]; // big!
  size_t n_wthor_filenames;

  // outputs
  char ids_file[PATH_MAX+1];
  char boards_file[PATH_MAX+1];
  char policy_file[PATH_MAX+1];

  // flags
  bool include_flips;
  // FIXME should the id behavior be configurable?
  // FIXME should we have any other variants of NN input? Is the valid move set helping?
  // FIXME should we have a flag for including/exluding duplicated board states
} config_t;

static config_t
load_config( char const * filename )
{
  toml_datum_t d;

  config_t ret;
  memset( &ret, 0, sizeof(ret) );

  FILE * config_file = fopen( filename, "r" );
  if( !config_file ) Fail( "Failed to open config file %s", filename );

  char errbuf[200]; memset( errbuf, 0, sizeof(errbuf) );
  toml_table_t const * toml_config = toml_parse_file( config_file, errbuf, sizeof(errbuf) );
  if( !toml_config ) Fail( "Failed to load toml file with %s", errbuf );

  toml_table_t const * files_table = toml_table_in( toml_config, "files" );
  if( !files_table ) Fail( "Expected [files] table at top level of config" );

  // -- inputs

  toml_array_t const * wthor_files = toml_array_in( files_table, "wthor_files" );
  if( !wthor_files ) Fail( "Expected [files.wthor_files] array" );
  if( toml_array_type( wthor_files ) != 's' ) Fail( "[files.wthor_files] array is not array of strings" );
  if( toml_array_nelem( wthor_files ) > CONFIG_FILES_MAX ) {
    Fail( "Cannot handle more than %d files, but got %d", CONFIG_FILES_MAX, toml_array_nelem( wthor_files ) );
  }

  for( int i = 0; i < toml_array_nelem( wthor_files ); ++i ) {
    d = toml_string_at( wthor_files, i );
    if( !d.ok ) Fail( "shouldn't happen" );
    if( strlen( d.u.s ) >= PATH_MAX ) Fail( "Filename %s is longer than PATH_MAX", d.u.s );
    memcpy( ret.wthor_filenames[i], d.u.s, strlen( d.u.s )+1 );
    free( d.u.s );
  }

  ret.n_wthor_filenames = (size_t)toml_array_nelem( wthor_files );

  // -- outputs

  d = toml_string_in( files_table, "ids_filename" );
  if( !d.ok ) Fail( "Expected files.ids_filename" );
  if( strlen( d.u.s ) > PATH_MAX ) Fail( "files.ids_filename too long" );

  memcpy( ret.ids_file, d.u.s, strlen(d.u.s) );
  free( d.u.s );

  d = toml_string_in( files_table, "boards_filename" );
  if( !d.ok ) Fail( "Expected files.boards_filename" );
  if( strlen( d.u.s ) > PATH_MAX ) Fail( "files.boards_filename too long" );

  memcpy( ret.boards_file, d.u.s, strlen(d.u.s) );
  free( d.u.s );

  d = toml_string_in( files_table, "policy_filename" );
  if( !d.ok ) Fail( "Expected files.policy_filename" );
  if( strlen( d.u.s ) > PATH_MAX ) Fail( "files.policy_filename too long" );

  memcpy( ret.policy_file, d.u.s, strlen(d.u.s) );
  free( d.u.s );

  toml_table_t * settings_table = toml_table_in( toml_config, "settings" );
  if( !settings_table ) Fail( "Expected [settings] table at top level of config" );

  toml_datum_t toml_include_flips = toml_bool_in( settings_table, "include_flips" );
  if( !toml_include_flips.ok ) Fail( "expected boolean at settings.include_flips" );

  ret.include_flips = toml_include_flips.u.b;

  toml_free( (void*)toml_config );
  fclose( config_file );
  return ret;
}

// ---------------------------
// output managment

typedef struct {
  zstd_file_t * ids;
  zstd_file_t * input;
  zstd_file_t * policy;
} outputs_t;

static outputs_t
outputs_from_config( config_t const * config )
{
  outputs_t outputs;

  outputs.ids = zstd_file_writer( config->ids_file );
  if( !outputs.ids ) Fail( "Failed to open board ids file at %s", config->ids_file );

  outputs.input = zstd_file_writer( config->boards_file ); // FIXME rationalize names
  if( !outputs.input ) Fail( "Failed to open boards file at %s", config->boards_file );

  outputs.policy = zstd_file_writer( config->policy_file );
  if( !outputs.policy ) Fail( "Failed to open policy file at %s", config->policy_file );

  return outputs;
}

static void
outputs_close( outputs_t * outputs )
{
  zstd_file_writer_close( outputs->ids );
  zstd_file_writer_close( outputs->input );
  zstd_file_writer_close( outputs->policy );
}

static void
outputs_save_game( outputs_t const *          outputs,
                   uint64_t                   id,
                   othello_game_t const *     game,
                   othello_move_ctx_t const * ctx,
                   uint8_t                    move_x,
                   uint8_t                    move_y )
{
  /* We produce three output files:
     1. Game ID file, used to select train/test split across _games_ not just board states
     2. The NN input file, contains the game repr that is fed into the NN
     3. The policy output, expected output from the NN */

  float input[193] = { 0 };
  float policy[64] = { 0 };

  nn_format_input( game, ctx, input );
  policy[move_x + move_y*8] = 1.0;

  zstd_file_writer_write( outputs->ids,    (void*)&id, sizeof(id) );
  zstd_file_writer_write( outputs->input,  (void*)input, sizeof(input) );
  zstd_file_writer_write( outputs->policy, (void*)policy, sizeof(policy) );
}

/// -----------------------------
/// flipperoo

static void
flip_full_game( othello_game_t const * game,
                uint8_t                move_x,
                uint8_t                move_y,
                uint8_t *              out_flipped_move_x,
                uint8_t *              out_flipped_move_y,
                othello_game_t *       out_flipped_game,
                othello_move_ctx_t *   out_flipped_ctx )
{

  memset( out_flipped_game, 0, sizeof(*out_flipped_game) );

  // -- setup move
  *out_flipped_move_x = 7 - move_x;
  *out_flipped_move_y = 7 - move_y;

  // -- setup game

  out_flipped_game->curr_player = game->curr_player;

  for( uint64_t x = 0; x < 8; ++x ) {
    for( uint64_t y = 0; y < 8; ++y ) {
      uint64_t flipx = 7 - x;
      uint64_t flipy = 7 - y;

      if( game->white & othello_bit_mask( x, y ) ) {
        // should be no piece here yet in new board
        assert( (out_flipped_game->white & othello_bit_mask( flipx, flipy )) ==0 );
        assert( (out_flipped_game->black & othello_bit_mask( flipx, flipy )) ==0 );

        out_flipped_game->white |= othello_bit_mask( flipx, flipy );
      }

      if( game->black & othello_bit_mask( x, y ) ) {
        // should be no piece here yet in new board
        assert( (out_flipped_game->white & othello_bit_mask( flipx, flipy )) ==0 );
        assert( (out_flipped_game->black & othello_bit_mask( flipx, flipy )) ==0 );

        out_flipped_game->black |= othello_bit_mask( flipx, flipy );
      }
    }
  }

  // -- setup ctx
  uint8_t winner;
  if( !othello_game_start_move( out_flipped_game, out_flipped_ctx, &winner ) ) {
    Fail( "failed to start flip game move" );
  }

  // sanity check
  // copy to avoid mutating
  othello_game_t     _tmp_game = *out_flipped_game;
  othello_move_ctx_t _tmp_ctx  = *out_flipped_ctx;

  if( !othello_game_make_move( &_tmp_game, &_tmp_ctx, othello_bit_mask( *out_flipped_move_x, *out_flipped_move_y ) ) ) {
    Fail( "Invalid flipped moved saved to file" );
  }
}

/// --------------------
/// actual wthor logic

static wthor_file_t const *
mmap_file( char const * fname )
{
  int fd = open( fname, O_RDONLY );
  if( fd==-1 ) Fail( "Failed to open %s", fname );

  wthor_header_t hdr[1];
  if( sizeof(hdr)!=read( fd, hdr, sizeof(hdr) ) ) {
    Fail( "Failed to read header from file %s", strerror( errno ) );
  }

  /* 0|8 -> 8x8
     10 -> 10x10 */

  if( hdr->p1 != 0 && hdr->p1 != 8 ) {
    Fail( "Unexpected value in header" );
  }

  size_t sz = sizeof( wthor_game_t ) * hdr->n1;
  void * mem = mmap( NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0 );
  if( mem==MAP_FAILED ) {
    Fail( "Failed to mmap with %s", strerror( errno ) );
  }

  close( fd );
  return mem;
}

static uint64_t
run_all_games_in_file( config_t const *     config,
                       outputs_t const *    outputs,
                       uint64_t             starting_id,
                       wthor_file_t const * file )
{
  for( size_t game_idx = 0; game_idx < wthor_file_n_games( file ); ++game_idx ) {
    wthor_game_t const * fgame = file->games + game_idx;

    /* Run the game, saving each board state and the move to take. */

    othello_game_t game[1];
    othello_game_init( game );

    /* Run the game, saving each state _and_ the move that was taken */

    uint8_t winner = (uint8_t)-1;
    for( size_t move_idx=0;; ++move_idx ) {
      othello_move_ctx_t ctx[1];
      if( !othello_game_start_move( game, ctx, &winner ) ) break;

      if( move_idx>=60 ) {
        /* Some of the games in the file seem to not have finished by the end of
           the fixed 60 move allotment. They do not have a winner as they are
           not complete? */
        break;
      }

      uint8_t move_byte  = fgame->moves[move_idx];

      uint8_t x, y;
      decode_move( move_byte, &x, &y );

      /* Some files have explicit pass byte */

      if( move_byte == 0 ) {
        /* Do not save PASS moves into the dataset */
        bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
        if( !valid ) Fail( "tried to make invalid move" );
        continue;
      }

      /* But not all of them.. If the known current player cannot make any
         moves, I guess they must have passed?

         Kinda irritating that we have to compute this move mask on every move
         so many times; maybe should restructure this? */

      if( ctx->n_own_moves == 0 ) {
        /* Do not save PASS moves into the dataset */

        bool valid = othello_game_make_move( game, ctx, OTHELLO_MOVE_PASS );
        if( !valid ) Fail( "tried to make invalid move" );
        /* do not continue, the current move applies to next player */
      }

      /* reset move ctx since we may have switched players */
      if( !othello_game_start_move( game, ctx, &winner ) ) Fail( "game should not be over" );

      outputs_save_game( outputs, starting_id, game, ctx, x, y );

      if( config->include_flips ) {
        othello_game_t     flipped_game[1];
        othello_move_ctx_t flipped_ctx[1];
        uint8_t            flipped_x, flipped_y;

        flip_full_game( game, x, y, &flipped_x, &flipped_y, flipped_game, flipped_ctx );

        /* Treat the flipped game as a unique game for the purposes of train/test split */
        starting_id += 1;
        outputs_save_game( outputs, starting_id, flipped_game, flipped_ctx, flipped_x, flipped_y );
      }

      bool valid = othello_game_make_move( game, ctx, othello_bit_mask( x, y ) );
      if( !valid ) {
        othello_board_print( game );
        Fail( "player %d tried to make invalid move at (%d, %d)", game->curr_player, x, y );
      }
    }

    starting_id += 1;
  }

  return starting_id;
}

int
main( int     argc,
      char ** argv )
{
  outputs_t outputs;
  config_t  config;

  if( argc != 2 ) Fail( "Usage: %s <config>", argv[0] );
  config = load_config( argv[1] );

  printf( "----------------------------\n" );
  printf( "Config loaded:\n" );
  printf( "  Num Input files: %zu\n", config.n_wthor_filenames );
  printf( "  Include flips: %s\n", config.include_flips ? "yes" : "no" );
  printf( "\n" );
  printf( "Outputs:\n" );
  printf( "  ids_file:    %s\n", config.ids_file );
  printf( "  boards_file: %s\n", config.boards_file );
  printf( "  policy_file: %s\n", config.policy_file );
  printf( "----------------------------\n" );

  outputs = outputs_from_config( &config );

  uint64_t id = 0;
  for( size_t file_idx = 0; file_idx < config.n_wthor_filenames; ++file_idx ) {
    printf( "On file %zu of %zu\n", file_idx, config.n_wthor_filenames );
    wthor_file_t const * file = mmap_file( config.wthor_filenames[file_idx] );
    id = run_all_games_in_file( &config, &outputs, id, file );
  }

  outputs_close( &outputs );

  return 0;
}
