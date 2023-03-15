#include "lib.h"
#include "../unit.h"

#include <stdlib.h>

TEST( fast_copy )
{
  uint8_t foo[1024];
  uint8_t bar[1024];

  for( size_t i = 0; i < 1024; ++i ) {
    foo[i] = (uint8_t)i;
  }

  for( size_t off = 0; off < 32; ++off ) {
    fast_copy( foo, bar+off, sizeof(foo)-off );
    CHECK_EQ( memcmp( foo, bar+off, sizeof(foo)-off ), 0 );
  }
}

int main()
{
  return unit_test_run_all( NULL );
}
