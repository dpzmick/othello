#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* Rules:
   - there are two players: white and black
   - board is 8x8 grid of cells
   - each cell contains a players piece, or is empty
   - Pieces must be placed such that there is a contiguous line between the new piece and another piece of the same color
   - When a piece is played, opposing pieces on the line(s) between the players piece are flipped over (capturing)
   - At least one piece must be captured must be flipped on each move
   - If a player cannot make a valid move, play passes to the next player.
   - When both players cannot make any moves, the game is over
   - The player with the most pieces on the board wins

   Using a bitboard, we need only 8*8=64 bits to store the occupancy state for each color.
*/

#define BIT_IDX(x,y)  ((y)*UINT64_C(8) + (x))
#define BIT_MASK(x,y) (UINT64_C(1) << BIT_IDX((x),(y)))

#define MIN(a,b) (a) < (b) ? (a) : (b)

typedef enum {
  COLOR_WHITE,
  COLOR_BLACK,
} color_t;

typedef struct {
  uint64_t white;
  uint64_t black;
} board_t;

void
board_zero( board_t * board )
{
  board->white = 0;
  board->black = 0;
}

void
board_init( board_t * board )
{
  board->white = BIT_MASK(3,3) | BIT_MASK(4,4);
  board->black = BIT_MASK(3,4) | BIT_MASK(4,3);
}

bool
board_check( board_t const * board )
{
  /* All cells must have exactly one color on them */
  return (board->white & board->black) == 0;
}

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

int
main( void )
{
  /* board_t board; */
  /* board_init( &board ); */
  /* board_print( &board ); */

  size_t cnt = 0;

  for( size_t y = 0; y<8; ++y ) { // for each row
    // find all horizontal lines
    for( size_t x1 = 0; x1<8; ++x1 ) {
      for( size_t x2 = 0; x2<8; ++x2 ) {
        if( x2<x1 ) continue;
        if( x2-x1>1 ) {
          printf("(%zu,%zu)->(%zu,%zu):\n\n", x1, y, x2, y);
          board_t board[1];
          board_zero( board );
          board->black |= BIT_MASK(x1,y) | BIT_MASK(x2,y);
          for( size_t x = x1+1; x<x2; ++x ) {
            board->white |= BIT_MASK(x,y);
          }
          board_print(board);
          printf("\n\n");
          cnt += 1;
        }
      }
    }
  }

  for( size_t x = 0; x<8; ++x ) { // for each col
    // find all vertical lines
    for( size_t y1 = 0; y1<8; ++y1 ) {
      for( size_t y2 = 0; y2<8; ++y2 ) { // FIXME this loop condition is dumb
        if( y2<y1 ) continue;
        if( y2-y1>1 ) {
          printf("(%zu,%zu)->(%zu,%zu):\n\n", x, y1, x, y2);
          board_t board[1];
          board_zero( board );
          board->black |= BIT_MASK(x,y1) | BIT_MASK(x,y2);
          for( size_t y = y1+1; y<y2; ++y ) {
            board->white |= BIT_MASK(x,y);
          }
          board_print(board);
          printf("\n\n");
          cnt += 1;
        }
      }
    }
  }

  /* diagonals are trickier, they either go left to right or right to left */
  /* start with left to right */

  for( size_t x = 0; x<8; ++x ) {
    for( size_t y = 0; y<8; ++y ) {
      // from (x,y), find all diagonal lines
      // note that "up" lines here would just be called right to left
      for( size_t d = 2; d<8; ++d ) { // must have at least one cell in the middle
        if( x+d >= 8 || y+d >=8 ) continue;

        printf("(%zu,%zu)->(%zu,%zu):\n\n", x, y, x+d, y+d);

        board_t board[1];
        board_zero( board );
        board->black = BIT_MASK(x,y) | BIT_MASK(x+d,y+d);
        for( size_t _d = 1; _d<d; ++_d ) {
          board->white |= BIT_MASK(x+_d,y+_d);
        }
        board_print(board);
        printf("\n\n");
        cnt += 1;
      }
    }
  }

  // diagonal lines going right are going "up"
  for( size_t x = 0; x<8; ++x ) {
    for( size_t y = 0; y<8; ++y ) {
      // from (x,y), find all diagonal lines
      for( size_t d = 2; d<8; ++d ) { // must have at least one cell in the middle
        if( x+d>=8 || d>y ) continue;

        printf("(%zu,%zu)->(%zu,%zu) (d=%zu):\n\n", x, y, x+d, y-d, d);

        board_t board[1];
        board_zero( board );
        board->black = BIT_MASK(x,y) | BIT_MASK(x+d,y-d);
        for( size_t _d = 1; _d<d; ++_d ) {
          board->white |= BIT_MASK(x+_d,y-_d);
        }
        board_print(board);
        printf("\n\n");
        cnt += 1;
      }
    }
  }

  printf("Total number of moves: %zu\n", cnt);
}
