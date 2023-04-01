#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

typedef struct wthor_header wthor_header_t;
typedef struct wthor_game   wthor_game_t;
typedef struct wthor_file   wthor_file_t;

struct __attribute__((packed)) wthor_header {
  uint8_t  century;
  uint8_t  year;
  uint8_t  month;
  uint8_t  day;
  uint32_t n1;
  uint16_t n2;
  uint16_t year_of_parties;     // ?
  uint8_t  p1;
  uint8_t  p2;
  uint8_t  p3;
  uint8_t  _pad;
};

static_assert( sizeof(wthor_header_t)==16, "Messed up size" );

struct __attribute__((packed)) wthor_game {
  uint64_t _junk;
  uint8_t  moves[60]; // always filled out
};

static_assert( sizeof(wthor_game_t)==68, "Messed up size" );

struct __attribute__((packed)) wthor_file {
  wthor_header_t hdr[1];
  wthor_game_t   games[];
};

static inline size_t
wthor_file_n_games( wthor_file_t const * file )
{
  return file->hdr->n1;
}

static inline size_t
wthor_file_mem_size( wthor_file_t const * file )
{
  return sizeof(wthor_header_t) + file->hdr->n1 * sizeof(wthor_game_t);
}

static inline void
decode_move( uint8_t   move,
             uint8_t * out_x,
             uint8_t * out_y )
{
  /* stored as column + (10*line) */

  // so a1 = 11 or x=0, y=0
  //    b1 = 21 or x=1, y=0

  *out_x = (move/10) - 1;
  *out_y = (move%10) - 1;
}
