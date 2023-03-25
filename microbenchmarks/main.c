#include "lib.h"

#include "pd_api.h"
#include <stdalign.h>
#include <stdbool.h>

static PlaydateAPI * G_pd      = NULL;
static float         G_freq_hz = 0;

static void
run_memcpy( void )
{
  size_t sz    = 4ul * 1024ul * 1024ul;
  size_t iters = 2;

  void * foo = G_pd->system->realloc( NULL, sz );
  if( !foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  void * bar = G_pd->system->realloc( NULL, sz );
  if( !bar ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    memcpy( foo, bar, sz );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  G_pd->system->realloc( bar, 0 );
  G_pd->system->realloc( foo, 0 );

  // read once, write once
  size_t bytes           = 2*sz*iters;
  float  mib             = (float)bytes/1024.0f/1024.0f;
  float  mib_per_sec     = mib/elapsed_sec;
  float  cycles          = elapsed_sec * G_freq_hz;
  float  bytes_per_cycle = ((float)bytes)/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", (double)mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", (double)bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );
  G_pd->system->realloc( buf, 0 );
}

static void
run_stream_copy( void )
{
  size_t sz    = 4ul * 1024ul * 1024ul;
  size_t iters = 8;

  float * _foo = G_pd->system->realloc( NULL, sz );
  if( !_foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _bar = G_pd->system->realloc( NULL, sz );
  if( !_bar ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * foo = (float*)align_ptr_up( (intptr_t)_foo, alignof(float) );
  float * bar = (float*)align_ptr_up( (intptr_t)_bar, alignof(float) );

  // shrink size to account for alignment that we had to shave off
  sz -= MAX( foo-_foo, bar-_bar );
  if( foo!=_foo || bar !=_bar ) {
    G_pd->system->logToConsole( "Got unaligned pointers %p and %p. Aligned to %p and %p. sz=%zu",
                                _foo, _bar, foo, bar, sz );
  }

  for( size_t i = 0; i < sz/4; ++i ) {
    foo[i] = i;
    bar[i] = 12;
  }

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    stream_copy( foo, bar, sz/sizeof(float) );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  if( 0!=memcmp( foo, bar, sz ) ) {
    char const * failed = "failed";
    G_pd->graphics->drawText( failed, strlen( failed ), kASCIIEncoding, 200, 40 );

    for( size_t i = 0; i < 4; ++i ) {
      G_pd->system->logToConsole( "[%d]: %f=%f", (int)i, (double)foo[i], (double)bar[i] );
    }
  }

  G_pd->system->realloc( _bar, 0 );
  G_pd->system->realloc( _foo, 0 );

  // read once, write once
  size_t bytes           = 2*sz*iters;
  float  mib             = (float)bytes/1024.0f/1024.0f;
  float  mib_per_sec     = mib/elapsed_sec;
  float  cycles          = elapsed_sec * G_freq_hz;
  float  bytes_per_cycle = ((float)bytes)/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", (double)mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );
  G_pd->system->realloc( buf, 0 ); // formatstring seems to leak if i reuse this

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", (double)bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "Elapsed: %f", (double)elapsed_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 120 );
  G_pd->system->realloc( buf, 0 ); // formatString seems to leak if I reuse this
}

static float scale_output[1024ul * 1024ul];
static float scale_q    = 3.14f;
static bool  scale_init = 0;

static void
run_stream_scale( void )
{
  if( !scale_init ) {
    for( size_t i = 0; i < ARRAY_SIZE( scale_output ); ++i ) {
      scale_output[i] = scale_q * (float)i;
    }
    scale_init = true;
  }

  size_t sz    = sizeof( scale_output );
  size_t iters = 8;

  // assume allocation is aligned

  float * foo = G_pd->system->realloc( NULL, sz );
  if( !foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * bar = G_pd->system->realloc( NULL, sz );
  if( !bar ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  for( size_t i = 0; i < ARRAY_SIZE( scale_output ); ++i ) {
    foo[i] = (float)i;
  }

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    stream_scale( foo, bar, scale_q, sz/sizeof(float) );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  G_pd->system->realloc( bar, 0 );
  G_pd->system->realloc( foo, 0 );

  if( 0!=memcmp( scale_output, bar, sz ) ) {
    char const * failed = "failed";
    G_pd->graphics->drawText( failed, strlen( failed ), kASCIIEncoding, 200, 40 );

    for( size_t i = sz/4-4; i < sz/4; ++i ) {
      G_pd->system->logToConsole( "[%d]: %f=%f", (int)i, (double)scale_output[i], (double)bar[i] );
    }
  }

  // read once, write once
  size_t bytes           = 2*sz*iters;
  float  mib             = (float)bytes/1024.0f/1024.0f;
  float  mib_per_sec     = mib/elapsed_sec;
  float  cycles          = elapsed_sec * G_freq_hz;
  float  bytes_per_cycle = ((float)bytes)/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", (double)mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );
  G_pd->system->realloc( buf, 0 ); // formatString seems to leak if I reuse this

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", (double)bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "Elapsed: %f", (double)elapsed_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 120 );
  G_pd->system->realloc( buf, 0 ); // formatString seems to leak if I reuse this
}

static void
run_stream_sum( void )
{
  static float sum_output[512ul * 1024ul];
  static bool  init    = 0;

  if( !init ) {
    for( size_t i = 0; i < ARRAY_SIZE( sum_output ); ++i ) {
      sum_output[i] = (float)i + (float)(i + 1);
    }
    init = true;
  }

  size_t sz    = sizeof(sum_output);
  size_t iters = 2;

  float * _foo = G_pd->system->realloc( NULL, sz );
  if( !_foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _bar = G_pd->system->realloc( NULL, sz );
  if( !_bar ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _baz = G_pd->system->realloc( NULL, sz );
  if( !_baz ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * foo = (float*)align_ptr_up( (intptr_t)_foo, alignof(float) );
  float * bar = (float*)align_ptr_up( (intptr_t)_bar, alignof(float) );
  float * baz = (float*)align_ptr_up( (intptr_t)_baz, alignof(float) );

  // shrink size to account for alignment that we had to shave off
  sz -= MAX( MAX( foo-_foo, bar-_bar ), baz-_baz );
  if( foo!=_foo || bar !=_bar || baz != _baz ) {
    G_pd->system->logToConsole( "Got unaligned pointers %p, %p, and %p. Aligned to %p, %p, and %p. sz=%zu",
                                _foo, _bar, _baz, foo, bar, _baz, sz );
  }

  for( size_t i = 0; i < sz/4; ++i ) {
    foo[i] = (float)i;
    bar[i] = (float)(i+1);
  }

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    stream_sum( foo, bar, baz, sz/sizeof(float) );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  G_pd->system->realloc( _baz, 0 );
  G_pd->system->realloc( _bar, 0 );
  G_pd->system->realloc( _foo, 0 );

  // read twice, write once
  size_t bytes           = 3*sz*iters;
  float  mib             = (float)bytes/1024.0f/1024.0f;
  float  mib_per_sec     = mib/elapsed_sec;
  float  cycles          = elapsed_sec * G_freq_hz;
  float  bytes_per_cycle = ((float)bytes)/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", (double)mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", (double)bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );
  G_pd->system->realloc( buf, 0 );
}

static void
run_stream_triad( void )
{
  size_t sz    = 2ul * 1024ul * 1024ul;
  size_t iters = 8;

  float * _foo = G_pd->system->realloc( NULL, sz );
  if( !_foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _bar = G_pd->system->realloc( NULL, sz );
  if( !_bar ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _baz = G_pd->system->realloc( NULL, sz );
  if( !_baz ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * foo = (float*)align_ptr_up( (intptr_t)_foo, alignof(float) );
  float * bar = (float*)align_ptr_up( (intptr_t)_bar, alignof(float) );
  float * baz = (float*)align_ptr_up( (intptr_t)_baz, alignof(float) );

  // shrink size to account for alignment that we had to shave off
  sz -= MAX( MAX( foo-_foo, bar-_bar ), baz-_baz );
  if( foo!=_foo || bar !=_bar || baz != _baz ) {
    G_pd->system->logToConsole( "Got unaligned pointers %p, %p, and %p. Aligned to %p, %p, and %p. sz=%zu",
                                _foo, _bar, _baz, foo, bar, _baz, sz );
  }

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    stream_triad( foo, bar, baz, 3.14f, sz/sizeof(float) );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  G_pd->system->realloc( _baz, 0 );
  G_pd->system->realloc( _bar, 0 );
  G_pd->system->realloc( _foo, 0 );

  // read twice, write once
  float  ops             = sz/sizeof(float)*iters;
  float  mflops          = ops/1000.0f/1000.0f/elapsed_sec;
  size_t bytes           = 3*sz*iters;
  float  mib             = (float)bytes/1024.0f/1024.0f;
  float  mib_per_sec     = mib/elapsed_sec;
  float  cycles          = elapsed_sec * G_freq_hz;
  float  bytes_per_cycle = ((float)bytes)/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", (double)mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", (double)bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "mflops: %0.3f", (double)mflops );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 120 );
  G_pd->system->realloc( buf, 0 );
}

static void
run_stream_triad_small( void )
{
  // cache is 8kib
  // we want whole data to fit there
  // and we're making 3 arrays
  // make sure they fit by dividing by 4

  size_t sz    = (8ul * 1024ul) / 4ul;
  size_t iters = 8192;

  float * _foo = G_pd->system->realloc( NULL, sz );
  if( !_foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _bar = G_pd->system->realloc( NULL, sz );
  if( !_bar ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * _baz = G_pd->system->realloc( NULL, sz );
  if( !_baz ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  float * foo = (float*)align_ptr_up( (intptr_t)_foo, alignof(float) );
  float * bar = (float*)align_ptr_up( (intptr_t)_bar, alignof(float) );
  float * baz = (float*)align_ptr_up( (intptr_t)_baz, alignof(float) );

  // shrink size to account for alignment that we had to shave off
  sz -= MAX( MAX( foo-_foo, bar-_bar ), baz-_baz );
  if( foo!=_foo || bar !=_bar || baz != _baz ) {
    G_pd->system->logToConsole( "Got unaligned pointers %p, %p, and %p. Aligned to %p, %p, and %p. sz=%zu",
                                _foo, _bar, _baz, foo, bar, _baz, sz );
  }

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    stream_triad( foo, bar, baz, 3.14f, sz/sizeof(float) );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  G_pd->system->realloc( _baz, 0 );
  G_pd->system->realloc( _bar, 0 );
  G_pd->system->realloc( _foo, 0 );

  // read twice, write once
  float  ops             = sz/sizeof(float)*iters;
  float  mflops          = ops/1000.0f/1000.0f/elapsed_sec;
  size_t bytes           = 3*sz*iters;
  float  mib             = (float)bytes/1024.0f/1024.0f;
  float  mib_per_sec     = mib/elapsed_sec;
  float  cycles          = elapsed_sec * G_freq_hz;
  float  bytes_per_cycle = ((float)bytes)/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", (double)mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", (double)bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );
  G_pd->system->realloc( buf, 0 );

  G_pd->system->formatString( &buf, "mflops: %0.3f", (double)mflops );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 120 );
  G_pd->system->realloc( buf, 0 );
}

static void
run_dual_issue( void )
{
  size_t iters = 1ul << 21ul;

  G_pd->system->resetElapsedTime();
  for( size_t i = 0; i < iters; ++i ) {
    run_128ins();
  }
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  float ops_per_sec   = (((float)iters)/elapsed_sec) * 128;
  float ops_per_cycle = ops_per_sec / G_freq_hz;

  /* char * buf; */

  /* G_pd->system->formatString( &buf, "Elapsed %0.3f", (double)elapsed_sec ); */
  /* G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 ); */
  /* G_pd->system->realloc( buf, 0 ); */

  /* G_pd->system->formatString( &buf, "Clock: %0.3f", (double)G_freq_hz ); */
  /* G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 ); */
  /* G_pd->system->realloc( buf, 0 ); */

  /* G_pd->system->formatString( &buf, "ops per cycle %0.3f", (double)ops_per_cycle ); */
  /* G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 120 ); */
  /* G_pd->system->realloc( buf, 0 ); */

  G_pd->system->logToConsole( "elapsed %0.3fs, iters: %d ops/c: %0.3f",
                              (double)elapsed_sec,
                              iters,
                              (double)ops_per_cycle );
}

static void
run_stride( void )
{
  const static size_t sz = 1024ul * 1024ul;

  uint8_t * foo = G_pd->system->realloc( NULL, sz );
  if( !foo ) {
    G_pd->system->logToConsole( "Allocation failed" );
    return;
  }

  memset( foo, 0, sz );

  const static size_t n_res = 256; // max I can support
  float results[n_res];
  size_t result_idx = 0;

  memset( results, 0, sizeof(results) );

  // we want to determine how much data we load "for free"
  // after we load from some part of an array.
  //
  // if our reads stay in the cache, they should be much faster

  size_t stride = 1;
  while( 1 ) {
    G_pd->system->resetElapsedTime();
    asm volatile( "dsb SY" ::: "memory" );
    stride_test( foo, sz, stride );
    asm volatile( "dsb SY" ::: "memory" );
    float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

    /* results[result_idx++] = sz/elapsed_sec/1024.0f/1024.0f; */
    results[result_idx++] = elapsed_sec;

    G_pd->system->logToConsole( "stride: %d, elapsed: %f sec, bw: %f MiB/s", (int)stride, (double)elapsed_sec, (double)results[result_idx-1] );

    stride += 1;

    if( stride > 256 ) break;
  }

  G_pd->system->realloc( foo, 0 );

  float max = 0;
  for( size_t i = 0; i < result_idx; ++i ) {
    max = MAX( results[i], max );
  }

  size_t height = 160;
  size_t width  = 340;
  float  x_off  = (float)width/result_idx;
  G_pd->graphics->drawRect( 20, 40, width, height, kColorBlack );

  for( size_t i = 0; i < result_idx; ++i ) {
    float scale = (float)height / (float)max;
    float p = -scale * results[i];

    G_pd->graphics->fillTriangle( /* x1 */ 20 + i*x_off - 2,
                                  /* y1 */ 40+height + p - 2,
                                  /* x2 */ 20 + i*x_off + 2,
                                  /* y2 */ 40+height + p - 2,
                                  /* x3 */ 20 + i*x_off,
                                  /* y3 */ 40+height + p + 2,
                                  kColorBlack );
  }
}

uint32_t G_selected = 6;

static int
update( void * usr )
{
  G_pd->graphics->clear( kColorWhite );

  // fetch any events
  PDButtons curr, pressed, released;;
  G_pd->system->getButtonState( &curr, &pressed, &released );

  if( kButtonLeft & pressed ) {
    G_selected = G_selected>0 ? G_selected-1 : G_selected;
  }

  if( kButtonRight & pressed ) {
    G_selected += 1;
  }

  // display a menu and run the selected test
  uint32_t x_sz  = 40;
  uint32_t x_off = 0;

  G_pd->graphics->drawText( "cpy", strlen( "cpy" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "cpy", strlen( "cpy" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "scl", strlen( "scl" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "sum", strlen( "sum" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "tri", strlen( "tri" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "trs", strlen( "trs" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "dul", strlen( "dul" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "str", strlen( "str" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawLine( G_selected*x_sz, 18, G_selected*x_sz+x_sz - 4, 18, 2, kColorBlack );

  switch( G_selected ) {
    case 0: run_memcpy(); break;
    case 1: run_stream_copy(); break;
    case 2: run_stream_scale(); break;
    case 3: run_stream_sum(); break;
    case 4: run_stream_triad(); break;
    case 5: run_stream_triad_small(); break;
    case 6: run_dual_issue(); break;
    case 7: run_stride(); break;
  }

  return 1;
}

static float
__attribute__((noinline))
estimate_clock_hz( void )
{
#ifdef TARGET_PLAYDATE
  // cannot get low level access to the cycle counter from DWT
  // because _it seems like_ the MPU is restricting those addresses
  //
  // have to use the playdate apis (which uses DWT->CYCCNT under the hood) to
  // try and estimate this. Kinda backwards b.c. they probably estimated clock
  // speed to do their cyccnt->time translation...

  // we expect that the cpu is running at around 186 MHz, which is ~5ns per
  // cycle.
  //
  // We can only measure in uS. With 1second of instructions, which
  // is 1e9/5 instruction, I'm seeing a clock estimate of 161 MHz. That's
  // probably pretty accurate. Same answer with the msec precision timer also
  // offered by playdate sdk

  uint32_t cnt = 500000000;

  /* uint32_t st_ms = G_pd->system->getCurrentTimeMilliseconds(); */
  G_pd->system->resetElapsedTime();

  // make sure all instructions are in the pipe. important?
  asm volatile( "isb" ); // not smart enough to compile with the intrinsics available

  // manually unroll the loop to make sure we're doing mostly just the add
  // instructions
  uint32_t a = 0;
  /* uint32_t b = 0; */
  /* uint32_t c = 0; */
  /* uint32_t d = 0; */
  for( uint32_t i = 0; i < cnt/128; ++i ) {

    // add should have 1 cycle of latency until result is ready
    // so adding into same register should consume the cycle
#define INS4                                              \
    asm volatile( "add %0, %0, 123" : "+r"(a) : : "cc" ); \
    asm volatile( "add %0, %0, 123" : "+r"(a) : : "cc" ); \
    asm volatile( "add %0, %0, 123" : "+r"(a) : : "cc" ); \
    asm volatile( "add %0, %0, 123" : "+r"(a) : : "cc" );

#define INS16 INS4  INS4  INS4  INS4
#define INS64 INS16 INS16 INS16 INS16
#define INS128 INS64 INS64

    INS128

#undef INS128
#undef INS64
#undef INS16
#undef INS4
  }

  // make sure all instructions are in the pipe. important?
  asm volatile( "isb" ); // not smart enough to compile with the intrinsics available

  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds
  return (float)cnt / elapsed_sec;

  /* uint32_t ed_ms = G_pd->system->getCurrentTimeMilliseconds(); */
  /* uint32_t elapsed_ms = (ed_ms-st_ms); */

  /* return (float)cnt / ((float)elapsed_ms * 0.001f); */

#else
  return 160.0f * 1000.0f * 1000.0f;
#endif
}

int
eventHandler( PlaydateAPI*  playdate,
              PDSystemEvent event,
              uint32_t      arg )
{
  (void)arg;

  if( event == kEventInit ) {
    G_pd = playdate;

    G_freq_hz = estimate_clock_hz();
    G_pd->system->logToConsole( "freq_mhz: %f\n", (double)(G_freq_hz * 0.001f * 0.001f) );

    /* Update callback is called at the speed required to hit refresh rate.

       Note that we can still take 10 seconds to run the update callback
       regardless of the configured refresh rate. */

    G_pd->display->setRefreshRate( 60.0 );
    G_pd->system->setUpdateCallback( update, NULL );
  }

  return 0;
}


// notes and references:
// - https://www.quinapalus.com/cm7cycles.html (sort of close to what we have)
