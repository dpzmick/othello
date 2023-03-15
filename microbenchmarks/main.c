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
  size_t iters = 4;

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

  float mib_per_sec     = ((float)(sz*iters))/elapsed_sec/1024.0f/1024.0f;
  float cycles          = elapsed_sec * G_freq_hz;
  float bytes_per_cycle = ((float)(sz*iters))/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );

  G_pd->system->realloc( buf, 0 );
}

static void
run_fast_copy( void )
{
  size_t sz    = 4ul * 1024ul * 1024ul;
  size_t iters = 4;

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
    fast_copy( foo, bar, sz );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  G_pd->system->realloc( bar, 0 );
  G_pd->system->realloc( foo, 0 );

  float mib_per_sec     = ((float)(sz*iters))/elapsed_sec/1024.0f/1024.0f;
  float cycles          = elapsed_sec * G_freq_hz;
  float bytes_per_cycle = ((float)(sz*iters))/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );

  G_pd->system->realloc( buf, 0 );
}

static void
run_stream_copy( void )
{
  size_t sz    = 4ul * 1024ul * 1024ul;
  size_t iters = 4;

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

  goto quit;

  G_pd->system->resetElapsedTime();
  asm volatile( "dsb SY" );
  for( size_t i = 0; i < iters; ++i ) {
    stream_copy( foo, bar, sz );
  }
  asm volatile( "dsb ST" );
  float elapsed_sec = G_pd->system->getElapsedTime(); // returns float seconds

  quit:

  G_pd->system->realloc( _bar, 0 );
  G_pd->system->realloc( _foo, 0 );
  return;

  float mib_per_sec     = ((float)(sz*iters))/elapsed_sec/1024.0f/1024.0f;
  float cycles          = elapsed_sec * G_freq_hz;
  float bytes_per_cycle = ((float)(sz*iters))/((float)cycles);

  char * buf;

  G_pd->system->formatString( &buf, "MiB/s: %0.3f", mib_per_sec );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 40 );

  G_pd->system->formatString( &buf, "bytes/cycle: %0.3f", bytes_per_cycle );
  G_pd->graphics->drawText( buf, strlen( buf ), kASCIIEncoding, 0, 80 );

  G_pd->system->realloc( buf, 0 );
}

uint32_t G_selected = 0;

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
  uint32_t x_sz  = 75;
  uint32_t x_off = 0;

  G_pd->graphics->drawText( "memcpy", strlen( "memcpy" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "fast_cpy", strlen( "fast_cpy" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawText( "strm_cpy", strlen( "strm_cpy" ), kASCIIEncoding, x_off, 0 );
  x_off += x_sz;

  G_pd->graphics->drawLine( G_selected*x_sz, 18, G_selected*x_sz+x_sz - 4, 18, 2, kColorBlack );

  // run the test
  if( G_selected==0 ) {
    run_memcpy();
  }

  if( G_selected==1 ) {
    run_fast_copy();
  }

  if( G_selected==2 ) {
    run_stream_copy();
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
  asm volatile( "isb 15" ); // not smart enough to compile with the intrinsics available

  // manually unroll the loop to make sure we're doing mostly just the add
  // instructions
  uint32_t foo = 0;
  for( uint32_t i = 0; i < cnt/50; ++i ) {
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
    asm volatile( "add %0, %0, 123" : "+r"(foo) : : "cc" );
  }

  // make sure all instructions are in the pipe. important?
  asm volatile( "isb 15" ); // not smart enough to compile with the intrinsics available

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
    G_pd->system->logToConsole( "freq_mhz: %f\n", G_freq_hz * 0.001f * 0.001f );

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
