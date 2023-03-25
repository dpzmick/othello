#include "lib.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(TARGET_PLAYDATE)
#include "pd_api.h"
#endif

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
             size_t                 n )
{
  for( size_t i = 0; i < n; ++i ) {
    b[i] = a[i];
  }
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

// Summing will required reading _two_ concurrent input streams
// And writing one output.
//
// So vs 1 load and 1 store, we are now doing 2 loads and 1 store per loop.
//
// We should expect to see a performance hit roughly 1/3 of 0.067, so maybe
// 0.022ish bytes per cycle
//
// But I'm getting 0.044, which doesn't make sense.
// Does this mean there's more speed to be had in the previous?
//
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

  // I've learned my lesson on trying to speed these up
  // just gonna leave this alone
}

// Disassembly of section .text.stream_triad:
// 60001740 <stream_triad>:
// 60001740:       b15b            cbz     r3, 6000175a <stream_triad+0x1a>
// 60001742:       eb00 0383       add.w   r3, r0, r3, lsl #2
// 60001746:       ecf0 7a01       vldmia  r0!, {s15}
// 6000174a:       ecb1 7a01       vldmia  r1!, {s14}
// 6000174e:       4298            cmp     r0, r3
// 60001750:       eee7 7a00       vfma.f32        s15, s14, s0
// 60001754:       ece2 7a01       vstmia  r2!, {s15}
// 60001758:       d1f5            bne.n   60001746 <stream_triad+0x6>
// 6000175a:       4770            bx      lr


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

void
stride_test( uint8_t * restrict a,
             size_t             sz,
             size_t             stride )
{
  for( size_t i = 0; i < sz; i += stride ) {
    a[i] = a[i] + a[i];
  }
}

// try doing a triad, but do an fma and an add each loop
// if this runs as fast as the cache-hit triad
// then we know we have dual issue

void
run_128ins( void )
{
#ifdef TARGET_PLAYDATE
  // 12 general purpose registers
  uint32_t foo1 = 1;
  uint32_t foo2 = 1; // issuing two adds per loop seems to work
  uint32_t foo3 = 1;
  uint32_t foo4 = 1;
  uint32_t foo5 = 1;
  uint32_t foo6 = 1;
  uint32_t foo7 = 1;
  uint32_t foo8 = 1;

  // 32 float registers (16 "lanes")
  float    s1   = 0.0f;
  float    s2   = 0.0f;
  float    s3   = 0.0f;
  float    s4   = 0.0f;
  float    s5   = 0.0f;
  float    s6   = 0.0f;
  float    s7   = 0.0f;
  float    s8   = 0.0f;

  // 1.9ish ins/cycle. feels dual issue 1 cycle result latency
/* #define INS4                                                \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo2) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo3) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo4) : : "cc" );  \ */
/* #define INS8 INS4 INS4 */
/* #define INS16 INS8 INS8 */

  // should be close to 1 ins/cycle, but I'm only getting 0.84ish
/* #define INS8                                                \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s1) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s2) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s3) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s4) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s5) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s6) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s7) :: "cc" ); \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s8) :: "cc" ); \ */
/* #define INS16 INS8 INS8 */

  // 1.17 ins/cycle
  // using dependent adds so they take one cycle
  // but maybe stalling b.c. of only 3 adds?
/* #define INS16                                               \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s1) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s2) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s3) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s4) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */

  // this only gets 1.3ish ops/sec
/* #define INS16                                               \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s1) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s2) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s3) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s4) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s5) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s6) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s7) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */
/*   asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s8) :: "cc" ); \ */
/*   asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \ */


  // 1.5 ins/cycle
#define INS32                                               \
  asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s1) :: "cc" ); \
  asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo2) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo3) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo4) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo5) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo6) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo7) : : "cc" );  \
  asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s2) :: "cc" ); \
  asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo2) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo3) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo4) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo5) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo6) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo7) : : "cc" );  \
  asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s3) :: "cc" ); \
  asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo2) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo3) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo4) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo5) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo6) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo7) : : "cc" );  \
  asm volatile( "vfma.f32 %0, %0, %0" : "+w"(s4) :: "cc" ); \
  asm volatile( "add %0, %0, 123" : "+r"(foo1) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo2) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo3) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo4) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo5) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo6) : : "cc" );  \
  asm volatile( "add %0, %0, 123" : "+r"(foo7) : : "cc" );  \

//#define INS32 INS16 INS16
#define INS64 INS32 INS32
#define INS128 INS64 INS64

      INS128

#endif
}
