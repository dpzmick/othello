#include "../libcommon/common.h"
#include "../libunit/unit.h"

#include <stdbool.h>
#include <stdint.h>

TEST( sanity )
{
  CHECK_EQ( 1, 1 );
}

TEST( keep_ith_bit )
{
  uint64_t bitset;
  uint64_t ret;

  bitset = 0b1;
  ret = keep_ith_set_bit( bitset, 0 );
  CHECK_EQ( ret, 0b1 );

  bitset = 0b1001;
  ret = keep_ith_set_bit( bitset, 0 );
  CHECK_EQ( ret, 0b1000 );

  bitset = 0b1001;
  ret = keep_ith_set_bit( bitset, 1 );
  CHECK_EQ( ret, 0b1 );

  bitset = 0b10101;
  ret = keep_ith_set_bit( bitset, 0 );
  CHECK_EQ( ret, 0b10000 );

  bitset = 0b10101;
  ret = keep_ith_set_bit( bitset, 1 );
  CHECK_EQ( ret, 0b00100 );

  bitset = 0b10101;
  ret = keep_ith_set_bit( bitset, 2 );
  CHECK_EQ( ret, 0b00001 );
}

/* Make sure we don't get tripped up by any funky integer promotion rules */
TEST( keep_ith_bit_64 )
{
  uint64_t bitset;
  uint64_t ret;

  bitset = 0x102004080000UL;
  ret = keep_ith_set_bit( bitset, 0 );
  CHECK_EQ( ret, 0x100000000000UL );
}
