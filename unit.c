#include "unit.h"

#include <assert.h>
#include <stdbool.h>

TEST(sanity)
{
  assert(true);
}

int main()
{
  unit_test_run_all();
}
