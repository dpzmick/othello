#include "../libunit/unit.h"

int main( int argc, char ** argv )
{
  char const * filter = argc > 1 ? argv[1] : NULL;
  return unit_test_run_all( filter );
}
