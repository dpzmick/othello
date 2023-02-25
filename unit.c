#include "unit.h"

char const * unit_test_names[1024];
test_func_t  unit_tests[1024];
size_t       test_cnt = 0;

int
unit_test_run_all( char const * filter )
{
  bool   any_failed = false;
  size_t filter_len = filter ? strlen( filter ) : 0;

  for( size_t i = 0; i < test_cnt; ++i ) {
    char const * test_name     = unit_test_names[i];
    size_t       test_name_len = strlen( test_name );

    if( filter ) {
      if( filter_len > test_name_len ) continue;
      if( 0 != memcmp( test_name, filter, filter_len ) ) continue;
    }

    printf( "Running %s\n", test_name );
    bool test_failed = false;
    unit_tests[i](&test_failed);
    any_failed |= test_failed;
  }

  return (int)any_failed;
}
