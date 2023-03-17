#include "lib.h"

#include <stddef.h>
#include <stdint.h>

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

// this is the fastest copy because the vector instructions
// seem to be able to get more loads and stores going
// somehow
void
stream_copy2( float const * restrict a,
              float * restrict       b,
              size_t                 n )
{
#ifdef TARGET_PLAYDATE
  // FPU has 32 registers
  size_t i = 0;

// NOTE: the 32 register version should not be any faster than the 2 register
// version? The memory bus is 64 bit
//
// with all registers enabled, getting 13ish MiB/s
// with 4 registers only, getting 13.5ish MiB/s
// FIXME try with 2?
//
// feels like there's maybe more room to go here? but at least we're probably
// saturating the memory bus.
#if 0
  for(; i < n ; i += 32 ) {
    if( n-i > 32 ) break;

    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s0-s31}"
                  :
                  : [ptr] "r"(_a)
                  : "s0",  "s1",  "s2",  "s3",
                    "s4",  "s5",  "s6",  "s7",
                    "s8",  "s9",  "s10", "s11",
                    "s12", "s13", "s14", "s15",
                    "s16", "s17", "s18", "s19",
                    "s20", "s21", "s22", "s23",
                    "s24", "s25", "s26", "s27",
                    "s28", "s29", "s30", "s31",
                    "cc" );

    asm volatile( "vstmia %[ptr]!, {s0-s31}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }

  for(; i < n ; i += 16 ) {
    if( n-i > 16 ) break;

    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s0-s15}"
                  :
                  : [ptr] "r"(_a)
                  : "s0",  "s1",  "s2",  "s3",
                    "s4",  "s5",  "s6",  "s7",
                    "s8",  "s9",  "s10", "s11",
                    "s12", "s13", "s14", "s15",
                    "cc" );

    asm volatile( "vstmia %[ptr]!, {s0-s15}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }

  for(; i < n ; i += 8 ) {
    if( n-i > 8 ) break;

    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s0-s7}"
                  :
                  : [ptr] "r"(_a)
                  : "s0",  "s1",  "s2",  "s3",
                    "s4",  "s5",  "s6",  "s7",
                    "cc" );

    asm volatile( "vstmia %[ptr]!, {s0-s7}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }
#endif

  for(; i < n ; i += 4 ) {
    if( n-i > 4 ) break;

    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s0-s3}"
                  :
                  : [ptr] "r"(_a)
                  : "s0",  "s1",  "s2",  "s3",
                    "cc" );

    asm volatile( "vstmia %[ptr]!, {s0-s3}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }

  for(; i < n ; i += 2 ) {
    if( n-i > 2 ) break;

    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s0-s1}"
                  :
                  : [ptr] "r"(_a)
                  : "s0",  "s1",
                    "cc" );

    asm volatile( "vstmia %[ptr]!, {s0-s1}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }

  for( ; i < n; ++i ) {
    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s0}"
                  :
                  : [ptr] "r"(_a)
                  : "s0", "cc" );

    asm volatile( "vstmia %[ptr]!, {s0}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }

#endif
}

// this is the same speed as normal copy
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
  // NOTE: this is not any faster
  // so fpu cannot multiply faster than ram?
  // FIXME work through that and make sure these numbers all add up

#ifdef TARGET_PLAYDATE
  size_t i = 0;

  for(; i < n ; i += 2 ) {
    if( n-i > 2 ) break;

    float const * _a = a+i;
    float *       _b = b+i;

    asm volatile( "vldmia %[ptr]!, {s1-s2}"
                  :
                  : [ptr] "r"(_a)
                  : "s0",  "s1",
                    "cc" );

    asm volatile( "vmul.f32 %[q], s1, s1" :: [q] "w"(q) : "s0", "cc" );
    asm volatile( "vmul.f32 %[q], s2, s2" :: [q] "w"(q) : "s1", "cc" );

    asm volatile( "vstmia %[ptr]!, {s1-s2}"
                  :
                  : [ptr] "r"(_b)
                  : "memory", "cc" );
  }

  for( ; i < n; ++i ) {
    float const * _a = a+i;
    float *       _b = b+i;

    // compiler moves the cmp ahead of the str, why?

    *_b = q * *_a;
  }

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
