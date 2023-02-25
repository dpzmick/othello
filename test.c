#include "unit.h"

#include "bitboard.h"
//#include "util.h"
//#include "game_tree.h"

#include <assert.h>
#include <stdbool.h>

int main()
{
  return unit_test_run_all( NULL );
}

// FIXME finish up the wraparound tests. all the diagonals are missing them
