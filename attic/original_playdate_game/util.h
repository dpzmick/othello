#pragma once

#include "bitboard.h"

// utility functions for local testing only

static inline board_t
new_board_from_str( char const * str )
{
  uint64_t white = 0;
  uint64_t black = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      assert(*str == 'B' || *str == 'W' || *str == '.');
      switch( *str ) {
        case 'W': white |= BIT_MASK(x,y); break;
        case 'B': black |= BIT_MASK(x,y); break;
        case '.': break;
      }
      str++;
    }
  }

  return (board_t){ .white=white, .black=black };
}

static inline uint64_t
parse_moves( char const * moves )
{
  uint64_t ret = 0;

  for( size_t y = 0; y < 8; ++y ) {
    for( size_t x = 0; x < 8; ++x ) {
      switch( *moves++ ) {
        case '*':
          ret |= BIT_MASK(x,y);
          break;
      }
    }
  }

  return ret;
}

static inline uint64_t
bitboard_from_rows( uint8_t rows[8] )
{
  uint64_t ret = 0;
  for( size_t row = 0; row < 8; ++row ) {
    uint64_t shift = 64 - 8*(row+1);
    ret |= ((uint64_t)rows[row]) << shift;
  }
  return ret;
}

