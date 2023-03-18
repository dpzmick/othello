#include "lib.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(TARGET_PLAYDATE)
#include "pd_api.h"
#endif

// this isn't stricly the standard "stream" benchmark
// but the stdlib memcpy was beating a[i] = b[i]
// so trying to optimize a bit to better understand the
// architecture.
//
// this is tied with playdate stdlib
void
fast_copy( uint8_t const * restrict a,
           uint8_t * restrict       b,
           size_t                   n )
{
  uint8_t const * ed = a+n;

  // peel off the front until we are aligned
  while( a < ed ) {
    // if we are aligned, break
    if( ((size_t)a & 0x20) == 0 ) break;

    // else copy as uint8_t
    *b++ = *a++;
  }

  // do main body as "word" sized loads and stores
  while( a < ed && ed-a > 8 ) {
    uint32_t _a1 = *((uint32_t*)a);
    *((uint32_t*)b) = _a1;

    a += 4;
    b += 4;
  }

  // peel 32 bit chunks off
  // this is going to run on 32bit arm so a word is u32
  while( a < ed && ed-a > 4 ) {
    uint32_t _a = *((uint32_t*)a);
    *((uint32_t*)b) = _a;

    a += 4;
    b += 4;
  }

  // finish up any funky sized tail
  while( a < ed ) {
    *b++ = *a++;
  }
}

// could also try doing copies with DMA controller?
//
// could also try loading 4-8 values into registers then emitting the stores
// since the memory bus may be wider that 32bit/to take advantage of cache
// more effectively? need to read docs further

// proper "stream" will run with floats and none of the fancy setup code.
// assumes that the buffer is aligned
//
// this is a lil faster but must assume that the buffer is aligned
//
// Disassembly of section .text.stream_copy:
//
// 60000750 <stream_copy>:
// 60000750:       b13a            cbz     r2, 60000762 <stream_copy+0x12>
// 60000752:       eb00 0282       add.w   r2, r0, r2, lsl #2
// 60000756:       f850 3b04       ldr.w   r3, [r0], #4
// 6000075a:       4290            cmp     r0, r2
// 6000075c:       f841 3b04       str.w   r3, [r1], #4
// 60000760:       d1f9            bne.n   60000756 <stream_copy+0x6>
// 60000762:       4770            bx      lr
//

void
stream_copy( float const * restrict a,
             float * restrict       b,
             size_t                  n )
{
  for( size_t i = 0; i < n; ++i ) {
    b[i] = a[i];
  }
}

void
stream_copy2( float const * restrict a,
              float * restrict       b,
              size_t                 n,
              PlaydateAPI *          pd )
{
#ifndef TARGET_PLAYDATE
  fast_copy( (uint8_t const*)a, (uint8_t*)b, n*4 );
#else
  float const * const ed = a+n;

  /* pd->system->logToConsole( "start" ); */
  /* pd->system->logToConsole( "a=%p, ed=%p, b=%p", a, ed, b ); */

  /* Adding additional copies does not seem to make a difference, but using the
     fancy loop below does seem to help with copy performance by a _very_ small
     amount. */

#if 0
  while( 1 ) {
    if( ed-a < 2 ) break; // less than two elements?

    // using write-back to update a pointer
    asm volatile( "vldmia %[ptr]!, {s0, s1}"
                  : [ptr] "+r"(a)
                  :: "s0", "s1", "cc" );

    // likewise, using write-back to update b pointer
    asm volatile( "vstmia %[ptr]!, {s0, s1}"
                  : [ptr] "+r"(b)
                  :: "memory", "cc" );

    /* pd->system->logToConsole( "a=%p, ed=%p, b=%p", a, ed, b ); */
  }
#endif

  /* Disassembly of section .text.stream_copy2:

     600014d0 <stream_copy2>:
     600014d0:       eb00 0282       add.w   r2, r0, r2, lsl #2
     600014d4:       4290            cmp     r0, r2
     600014d6:       d805            bhi.n   600014e4 <stream_copy2+0x14>
     600014d8:       ecb0 0a01       vldmia  r0!, {s0}
     600014dc:       eca1 0a01       vstmia  r1!, {s0}
     600014e0:       4282            cmp     r2, r0
     600014e2:       d2f9            bcs.n   600014d8 <stream_copy2+0x8>
     600014e4:       4770            bx      lr
     600014e6:       bf00            nop

     Pretty clean?
   */

  while( 1 ) {
    if( a > ed ) break;

    // using write-back to update a pointer
    //
    // this asm technically doesn't need the volatile marker
    asm volatile( "vldmia %[ptr]!, {s0}"
                  : [ptr] "+r"(a)
                  :: "s0", "cc" );

    // likewise, using write-back to update b pointer
    //
    // but this one does require volatile, just plain "memory" isn't strong
    // enough to keep it in
    asm volatile( "vstmia %[ptr]!, {s0}"
                  : [ptr] "+r"(b)
                  :: "memory", "cc" );

    /* pd->system->logToConsole( "a=%p, ed=%p, b=%p", a, ed, b ); */
  }
#endif
}

// This should always run at the same speed as a normal copy.
//
// The naive direct implementation at O2 is getting 0.066 bytes per cycle.
// But the fastest copy I can do is getting 0.067 bytes per cycle!
//
// This is _tragic_ and must be futher optimized.
//
// Naive version compiles to:
//
// Disassembly of section .text.stream_scale:
// 60001440 <stream_scale>:
// 60001440:       b14a            cbz     r2, 60001456 <stream_scale+0x16>
// 60001442:       eb00 0282       add.w   r2, r0, r2, lsl #2
// 60001446:       ecf0 7a01       vldmia  r0!, {s15}
// 6000144a:       ee67 7a80       vmul.f32        s15, s15, s0
// 6000144e:       4290            cmp     r0, r2
// 60001450:       ece1 7a01       vstmia  r1!, {s15}
// 60001454:       d1f7            bne.n   60001446 <stream_scale+0x6>
// 60001456:       4770            bx      lr
//
//
// Which is very close to what we probably want

void
stream_scale( float const * restrict a,
              float * restrict       b,
              float                  q,
              size_t                 n )
{
  for( size_t i = 0; i < n; ++i ) {
    b[i] = q*a[i];
  }
}

void
stream_scale2( float const * restrict a,
               float * restrict       b,
               float                  q,
               size_t                 n )
{
#ifndef TARGET_PLAYDATE
  stream_scale( a, b, q, n );
#else

  // need to make sure that the fmul latency doesn't stall
  // our memory operations.
  // If we're doing
  //
  // load
  // mul
  // store
  //
  // we aren't getting the store started quickly enough to fully saturate memory
  //
  // if we instead can somehow do
  // load
  // store
  // mul
  // mul
  // ...
  //
  // so that the mul runs in parallel with the memory operations, we're in a
  // better spot.
  //
  // Knowing how far ahead we need to fill the pipeline is a bit tricky but this
  // seems to be working with a distance of 2. Ideally the instructio scheduler
  // would be doing this for us, but it seems not to be kicking in and getting
  // it.
  //
  // Compiler was unable to generate the code I wanted with a variety of coaxes,
  // so have to write this is psudeo-asm too.
  //
  // This code sits solidly at 0.067 bytes per cycle, assuming I calculated that
  // correctly. Is that good? No idea

  float const * ed = a+n; // last element

  // first load the top of the array into register s2,s3 (our "ahead" registers)
  // this instruction increments (a)
  asm volatile( "vldmia %[ptr]!, {s2,s3}"
                : [ptr] "+r"(a)
                :: "s2", "s3", "cc" );

  while( 1 ) {
    // check if we can load two more elements
    bool last_load = (ed-a) < 2;

    // multiply q*{s2,s3} -> {s0,s1}
    asm volatile( "vmul.f32 s0, %[q], s2"
                  :: [q] "w"(q)
                  : "s0", "s2", "cc" );

    asm volatile( "vmul.f32 s1, %[q], s3"
                  :: [q] "w"(q)
                  : "s1", "s3", "cc" );

    if( !last_load ) { // need space for two elements
      // load a->{s2,s3}
      asm volatile( "vldmia %[ptr]!, {s2, s3}"
                    : [ptr] "+r"(a)
                    :: "s2", "s3", "cc" );
    }

    // store {s0,s1}->b
    asm volatile( "vstmia %[ptr]!, {s0,s1}"
                  : [ptr] "+r"(b)
                  :: "memory", "cc" );

    if( last_load ) break; // done loading, s2,s3 left undefined
  }

  // finish off the processing with straightforward loop
  while( 1 ) {
    if( a > ed ) break;
    *b = q * *a;

    a += 1;
    b += 1;
  }

  // FIXME check the exit behavior for arrays not multiple of 2
#endif
}

// this is a bit slower that stream_copy
// so I guess adds are slowish?
//
// 60000770 <stream_sum>:
// 60000770:       b15b            cbz     r3, 6000078a <stream_sum+0x1a>
// 60000772:       eb00 0383       add.w   r3, r0, r3, lsl #2
// 60000776:       ecf0 7a01       vldmia  r0!, {s15}
// 6000077a:       ecb1 7a01       vldmia  r1!, {s14}
// 6000077e:       4298            cmp     r0, r3
// 60000780:       ee77 7a87       vadd.f32        s15, s15, s14
// 60000784:       ece2 7a01       vstmia  r2!, {s15}
// 60000788:       d1f5            bne.n   60000776 <stream_sum+0x6>
// 6000078a:       4770            bx      lr
//
void
stream_sum( float const * restrict a,
            float const * restrict b,
            float * restrict       c,
            size_t                 n )
{
  for( size_t i = 0; i < n; ++i ) {
    c[i] = a[i] + b[i];
  }
}

// compiles to:
//
// 600008a0 <stream_triad.constprop.0>:
// 600008a0:       b16b            cbz     r3, 600008be <stream_triad.constprop.0+0x1e>
// 600008a2:       eddf 6a07       vldr    s13, [pc, #28]  ; 600008c0 <stream_triad.constprop.0+0x20>
// 600008a6:       eb00 0383       add.w   r3, r0, r3, lsl #2
// 600008aa:       ecf0 7a01       vldmia  r0!, {s15}
// 600008ae:       ecb1 7a01       vldmia  r1!, {s14}
// 600008b2:       4298            cmp     r0, r3
// 600008b4:       eee7 7a26       vfma.f32        s15, s14, s13
// 600008b8:       ece2 7a01       vstmia  r2!, {s15}
// 600008bc:       d1f5            bne.n   600008aa <stream_triad.constprop.0+0xa>
// 600008be:       4770            bx      lr
// 600008c0:       4048f5c3        submi   pc, r8, r3, asr #11
//
// so we get the fma, as expected.
// seeing 1.7ish mflops
//
// same speed as the adds

void
stream_triad( float const * restrict a,
              float const * restrict b,
              float * restrict       c,
              float                  q,
              size_t                 n )
{
  for( size_t i = 0; i < n; ++i ) {
    c[i] = a[i] + q*b[i];
  }
}
