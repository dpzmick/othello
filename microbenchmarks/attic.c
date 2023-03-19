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

void
stream_scale2( float const * restrict a,
               float * restrict       b,
               float                  q,
               size_t                 n )
{
#ifndef TARGET_PLAYDATE
  stream_scale( a, b, q, n );
#else
  float const * ed = a+n; // last element

  // first load the top of the array into register s1 (our "ahead" register)
  // this instruction increments (a)
  asm volatile( "vldmia %[ptr]!, {s1}"
                : [ptr] "+r"(a)
                :: "s1", "cc" );

  while( 1 ) {
    // check if we can load two more elements
    bool last_load = a >= ed;

    // multiply q*s1 -> s0
    asm volatile( "vmul.f32 s0, %[q], s1"
                  :: [q] "w"(q)
                  : "s0", "cc" );

    if( !last_load ) {
      // load a->s1
      asm volatile( "vldmia %[ptr]!, {s1}"
                    : [ptr] "+r"(a)
                    :: "s1", "cc" );
    }

    // store s0->b
    asm volatile( "vstmia %[ptr]!, {s0}"
                  : [ptr] "+r"(b)
                  :: "memory", "cc" );

    if( last_load ) break; // done loading, s2,s3 left undefined
  }
#endif
}


void
stream_copy2( float const * restrict a,
              float * restrict       b,
              size_t                 n,
              PlaydateAPI *          pd )
{
#ifndef TARGET_PLAYDATE
  stream_copy( a, b, n );
#else
  float const * const ed = a+n;

  // This verision seems to very slightly beats the naive one some of the time.
  // Not entirely sure why.

  while( 1 ) {
    if( ed-a < 4 ) break;

    // using write-back to update a pointer
    asm volatile( "vldmia %[ptr]!, {s0, s1, s2, s3}"
                  : [ptr] "+r"(a)
                  :: "s0", "s1", "s2", "s3", "cc" );

    // likewise, using write-back to update b pointer
    asm volatile( "vstmia %[ptr]!, {s0, s1, s2, s3}"
                  : [ptr] "+r"(b)
                  :: "memory", "cc" );
  }

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
  }
#endif
}
