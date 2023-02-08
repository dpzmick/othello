#pragma once

#include <stdbool.h>
#include <stdio.h>

typedef void (*test_func_t)();

static char const * unit_test_names[1024];
static test_func_t  unit_tests[1024];
static size_t       test_cnt = 0;

#define TEST(name) \
  static void TEST_##name();                    \
                                                \
  __attribute__((constructor))                  \
  static void TEST_##name##_ctor() {            \
    unit_test_names[test_cnt] = #name;          \
    unit_tests[test_cnt++] = TEST_##name;       \
  }                                             \
                                                \
  static void TEST_##name( bool * __attribute__((unused)) __test_failed )

// from http://www.robertgamble.net/2012/01/c11-generic-selections.html
// but modified to printing bitboards (u64) in hex
#define printf_format(x) _Generic((x), \
    char: "%c", \
    signed char: "%hhd", \
    unsigned char: "%hhu", \
    signed short: "%hd", \
    unsigned short: "%hu", \
    signed int: "%d", \
    unsigned int: "%u", \
    long int: "%ld", \
    unsigned long int: "%lu", \
    long long int: "%lld", \
    unsigned long long int: "%llx", \
    float: "%f", \
    double: "%f", \
    long double: "%Lf", \
    char *: "%s", \
    void *: "%p")

/* print file:line so emacs picks this up as an error marker and we can hit
   enter to jump there */

#define CHECK_EQ(a,b)                                                   \
  do {                                                                  \
    if( (a)!=(b) ) {                                                    \
      *__test_failed = true;                                            \
      printf( "%s:%d: CHECK_EQ(%s, %s): ", __FILE__, __LINE__, #a, #b ); \
      printf( printf_format(a), a );                                    \
      printf( "!=" );                                                   \
      printf( printf_format(b), b );                                    \
      printf( "\n" );                                                   \
    }                                                                   \
  } while(0)

#define CHECK( b )                                                      \
  do {                                                                  \
    if( !(b) ) {                                                        \
      *__test_failed = true;                                            \
      printf( "%s:%d: CHECK(false)\n", __FILE__, __LINE__ );            \
    }                                                                   \
  } while(0)

static int
unit_test_run_all()
{
  bool any_failed = false;
  for( size_t i = 0; i < test_cnt; ++i ) {
    printf( "Running %s\n", unit_test_names[i]);
    bool test_failed = false;
    unit_tests[i](&test_failed);
    any_failed |= test_failed;
  }

  return (int)any_failed;
}
