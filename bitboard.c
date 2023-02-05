#include <stdio.h>

#include "bitboard.h"

void
board_print( board_t const * board )
{
  printf("  | 0 ");
  for( size_t x = 1; x < 8; ++x ) {
    printf("%zu ", x);
  }
  printf("\n--+----------------\n");

  for( size_t y = 0; y < 8; ++y ) {
    printf("%zu | ", y );

    for( size_t x = 0; x < 8; ++x ) {
      uint8_t bit_white = (board->white & BIT_MASK(x,y)) != 0;
      uint8_t bit_black = (board->black & BIT_MASK(x,y)) != 0;

      if( bit_white && bit_black ) {
        printf( "X " ); // invalid
      }
      else if( bit_white ) {
        printf( "W " );
      }
      else if( bit_black ) {
        printf( "B " );
      }
      else {
        printf( "  " );
      }
    }
    printf("\n");
  }
}
