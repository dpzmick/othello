#include "../libcommon/common.h"
#include "../libcommon/hash.h"
#include "../libcomputer/nn_policy.h"
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
#include <sys/stat.h>
#include <sys/types.h>
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
  char boards_dir[PATH_MAX+1];
  char policy_file[PATH_MAX+1];

  // flags
  int64_t boards_per_file;
  int64_t board_lookback;
  bool    include_flips;
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

  memcpy( ret.ids_file, d.u.s, strlen( d.u.s )+1 );
  free( d.u.s );

  d = toml_string_in( files_table, "boards_dir" );
  if( !d.ok ) Fail( "Expected files.boards_dir" );
  if( strlen( d.u.s ) > PATH_MAX ) Fail( "files.boards_dir too long" );

  memcpy( ret.boards_dir, d.u.s, strlen( d.u.s )+1 );
  free( d.u.s );

  d = toml_string_in( files_table, "policy_filename" );
  if( !d.ok ) Fail( "Expected files.policy_filename" );
  if( strlen( d.u.s ) > PATH_MAX ) Fail( "files.policy_filename too long" );

  memcpy( ret.policy_file, d.u.s, strlen( d.u.s )+1 );
  free( d.u.s );

  toml_table_t * settings_table = toml_table_in( toml_config, "settings" );
  if( !settings_table ) Fail( "Expected [settings] table at top level of config" );

  toml_datum_t toml_include_flips = toml_bool_in( settings_table, "include_flips" );
  if( !toml_include_flips.ok ) Fail( "expected boolean at settings.include_flips" );

  ret.include_flips = toml_include_flips.u.b;

  toml_datum_t toml_board_lookback = toml_int_in( settings_table, "board_lookback" );
  if( !toml_board_lookback.ok ) Fail( "expected in at settings.board_lookback" );
  if( toml_board_lookback.u.i < 0 ) Fail( "board lookback must be >= 0" );

  ret.board_lookback = toml_board_lookback.u.i;

  toml_datum_t toml_boards_per_file = toml_int_in( settings_table, "boards_per_file" );
  if( !toml_boards_per_file.ok ) Fail( "expected value for settings.boards_per_file" );
  if( toml_boards_per_file.u.i < 0 ) Fail( "invalid boards per file setting" );

  ret.boards_per_file = toml_boards_per_file.u.i;

  toml_free( (void*)toml_config );
  fclose( config_file );
  return ret;
}

// ---------------------------
// output managment

typedef struct {
  zstd_file_t * ids;
  zstd_file_t * policy;

  char          boards_dir[PATH_MAX+1];
  size_t        next_boards_file_idx;
  size_t        n_entries_in_file;
  size_t        entries_per_file;
  zstd_file_t * boards_file;
} outputs_t;

static void
outputs_open_next_boards_file( outputs_t * outputs );

static outputs_t
outputs_from_config( config_t const * config )
{
  outputs_t outputs;

  outputs.ids = zstd_file_writer( config->ids_file );
  if( !outputs.ids ) Fail( "Failed to open board ids file at %s", config->ids_file );

  outputs.policy = zstd_file_writer( config->policy_file );
  if( !outputs.policy ) Fail( "Failed to open policy file at %s", config->policy_file );

  if( 0 != mkdir( config->boards_dir, 0755 ) ) {
    if( errno != EEXIST ) {
      Fail( "Failed to mkdir %s", config->boards_dir );
    }
  }

  memcpy( outputs.boards_dir, config->boards_dir, strlen( config->boards_dir )+1 ); // size already checked
  outputs.next_boards_file_idx = 0;
  outputs.boards_file = NULL;
  outputs_open_next_boards_file( &outputs );

  outputs.entries_per_file = (size_t)config->boards_per_file;

  return outputs;
}

static void
outputs_close( outputs_t * outputs )
{
  zstd_file_writer_close( outputs->ids );
  zstd_file_writer_close( outputs->policy );
  zstd_file_writer_close( outputs->boards_file ); // should be non-null

  printf( "Closed last output board with %zu entries\n", outputs->n_entries_in_file );
}

static void
outputs_open_next_boards_file( outputs_t * outputs )
{
  if( outputs->boards_file ) zstd_file_writer_close( outputs->boards_file );

  char fname[PATH_MAX+32+1];
  sprintf( fname, "%s/%05zu.dat.zst", outputs->boards_dir, outputs->next_boards_file_idx );
  outputs->next_boards_file_idx += 1;

  outputs->boards_file = zstd_file_writer( fname );
  if( !outputs->boards_file ) Fail( "Failed to open boards file at %s", fname );

  printf( "Rolling output files, previous had %zu entries, next file is %s\n", outputs->n_entries_in_file, fname );
  outputs->n_entries_in_file = 0;
}

static void
outputs_save_game( outputs_t *                outputs,
                   uint64_t                   id,
                   othello_game_t const *     game,
                   othello_move_ctx_t const * ctx,
                   uint8_t                    move_x,
                   uint8_t                    move_y,
                   othello_game_t const *     history,
                   size_t                     n_history )
{
  /* We produce three output files:
     1. Game ID file, used to select train/test split across _games_ not just board states
     2. The NN input file, contains the game repr that is fed into the NN
     3. The policy output, expected output from the NN */

  size_t n = 64+128+(128*n_history);  /* canonicalized; no player byte */

  /* nn_format_input writes float32 0.0/1.0 values; downcast to uint8 so the
     on-disk boards file is 1/4 the size and the Python loader reads uint8
     directly. */
  float   input_f[n];           /* ugh more vlas */
  uint8_t input_u8[n];
  memset( input_f, 0, n*sizeof(float) );
  nn_format_input( game, ctx, history, n_history, input_f );
  for( size_t i = 0; i < n; ++i ) input_u8[i] = (uint8_t)input_f[i];

  /* NOTE: policy target is one-hot per row. Duplicate boards across games
     are written as separate rows rather than aggregated into a single soft
     target. Cross-entropy with N duplicate one-hots equals one soft-target
     row up to a 1/N scalar, so the model learns the empirical move
     distribution either way -- but the duplicates inflate compute and
     memory. Pre-aggregating boards with sample weights would cut training
     time substantially. notes.org lists "deduplicate board positions?" as
     a planned data-gen flag. */
  float policy[64] = { 0 };
  policy[move_x + move_y*8] = 1.0;

  zstd_file_writer_write( outputs->ids,    (void*)&id,    sizeof(id)     );
  zstd_file_writer_write( outputs->policy, (void*)policy, sizeof(policy) );

  if( outputs->n_entries_in_file >= outputs->entries_per_file ) {
    outputs_open_next_boards_file( outputs );
  }

  outputs->n_entries_in_file += 1;
  zstd_file_writer_write( outputs->boards_file, (void*)input_u8, sizeof(input_u8) );
}

/// -----------------------------
/// symmetry transformations

/* Full D4: 4 rotations + 4 reflections. The 8 symmetries together generate
   all dihedral transforms of the 8x8 board. Othello has full D4 symmetry --
   the starting position and the rules are invariant under any of these --
   so each WTHOR game contributes 8 distinct training samples.

   transform_id:
     0 = identity:        (x, y) -> (x, y)
     1 = rot 90 CCW:      (x, y) -> (y, 7-x)
     2 = rot 180:         (x, y) -> (7-x, 7-y)
     3 = rot 270 CCW:     (x, y) -> (7-y, x)
     4 = flip horizontal: (x, y) -> (7-x, y)
     5 = transpose:       (x, y) -> (y, x)
     6 = flip vertical:   (x, y) -> (x, 7-y)
     7 = anti-transpose:  (x, y) -> (7-y, 7-x)
*/
#define N_SYMMETRIES 8

static inline void
transform_xy( int transform_id, uint8_t x, uint8_t y, uint8_t * out_x, uint8_t * out_y )
{
  switch( transform_id ) {
    case 0: *out_x = x;     *out_y = y;     break;
    case 1: *out_x = y;     *out_y = 7 - x; break;
    case 2: *out_x = 7 - x; *out_y = 7 - y; break;
    case 3: *out_x = 7 - y; *out_y = x;     break;
    case 4: *out_x = 7 - x; *out_y = y;     break;
    case 5: *out_x = y;     *out_y = x;     break;
    case 6: *out_x = x;     *out_y = 7 - y; break;
    case 7: *out_x = 7 - y; *out_y = 7 - x; break;
    default: Fail( "bad transform_id %d", transform_id );
  }
}

static void
transform_full_game( int                    transform_id,
                     othello_game_t const * game,
                     uint8_t                move_x,
                     uint8_t                move_y,
                     uint8_t *              out_move_x,
                     uint8_t *              out_move_y,
                     othello_game_t *       out_game,
                     othello_move_ctx_t *   out_ctx )
{
  memset( out_game, 0, sizeof(*out_game) );

  transform_xy( transform_id, move_x, move_y, out_move_x, out_move_y );

  out_game->curr_player = game->curr_player;

  for( uint8_t x = 0; x < 8; ++x ) {
    for( uint8_t y = 0; y < 8; ++y ) {
      uint8_t tx, ty;
      transform_xy( transform_id, x, y, &tx, &ty );

      if( game->white & othello_bit_mask( x, y ) ) {
        out_game->white |= othello_bit_mask( tx, ty );
      }
      if( game->black & othello_bit_mask( x, y ) ) {
        out_game->black |= othello_bit_mask( tx, ty );
      }
    }
  }

  uint8_t winner;
  if( !othello_game_start_move( out_game, out_ctx, &winner ) ) {
    Fail( "transform t=%d produced a finished game", transform_id );
  }

  /* sanity: the transformed move should be valid on the transformed board */
  othello_game_t     _tmp_game = *out_game;
  othello_move_ctx_t _tmp_ctx  = *out_ctx;
  if( !othello_game_make_move( &_tmp_game, &_tmp_ctx, othello_bit_mask( *out_move_x, *out_move_y ) ) ) {
    Fail( "transformed move (t=%d) is not valid", transform_id );
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
                       outputs_t *          outputs,
                       uint64_t             starting_id,
                       wthor_file_t const * file )
{
  /* When include_flips is on we emit all 4 D4 rotations per board. The 180
     rotation maps Othello's starting position to itself; the 90/270 ones
     produce a color-swapped layout that nn_format_input absorbs via
     canonicalization. */
  int n_transforms = config->include_flips ? N_SYMMETRIES : 1;

  for( size_t game_idx = 0; game_idx < wthor_file_n_games( file ); ++game_idx ) {
    wthor_game_t const * fgame = file->games + game_idx;

    /* Run the game, saving each board state and the move to take. */

    othello_game_t game[1];
    othello_game_init( game );

    /* VLA again.. Why not. One history buffer per transform. */
    othello_game_t histories[N_SYMMETRIES][config->board_lookback];
    memset( histories, 0, sizeof(histories) );

    /* The id identifies the game (not the move) for train/test split purposes.
       All moves of the unflipped game share game_id; each rotation variant
       gets its own id. After the game we advance starting_id by the number
       of transforms we emitted. */

    uint64_t transform_ids[N_SYMMETRIES];
    for( int t = 0; t < n_transforms; ++t ) {
      transform_ids[t] = starting_id + (uint64_t)t;
    }

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

      /* Emit one training row per transform. t=0 is the identity, so the
         original (no-rotation) game is always saved. */
      for( int t = 0; t < n_transforms; ++t ) {
        othello_game_t     tgame[1];
        othello_move_ctx_t tctx[1];
        uint8_t            tx, ty;

        transform_full_game( t, game, x, y, &tx, &ty, tgame, tctx );

        /* Each transform variant is a unique "game" for the purposes of
           train/test split, but uses a single id across all of its moves. */
        outputs_save_game( outputs, transform_ids[t], tgame, tctx, tx, ty,
                           histories[t], (size_t)config->board_lookback );

        for( int64_t i = 1; i < config->board_lookback; ++i ) {
          histories[t][i-1] = histories[t][i];
        }
        if( config->board_lookback > 0 ) {
          histories[t][config->board_lookback-1] = *tgame;
        }
      }

      bool valid = othello_game_make_move( game, ctx, othello_bit_mask( x, y ) );
      if( !valid ) {
        othello_board_print( game );
        Fail( "player %d tried to make invalid move at (%d, %d)", game->curr_player, x, y );
      }
    }

    starting_id += (uint64_t)n_transforms;
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
  printf( "  Num Input files: %zu\n",  config.n_wthor_filenames );
  //printf( "  Boards per file: %ld\n",  config.boards_per_file );
  printf( "  Include flips:   %s\n",   config.include_flips ? "yes" : "no" );
  printf( "\n" );
  printf( "Outputs:\n" );
  printf( "  ids_file:    %s\n", config.ids_file );
  printf( "  boards_dir:  %s\n", config.boards_dir );
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
