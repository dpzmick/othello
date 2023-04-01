# removed -pedantic b.c. gcc extnsion for binary literals

CFLAGS+=                  \
  -std=c11                \
  -D_GNU_SOURCE           \
  -Werror                 \
  -Wall                   \
  -Wextra                 \
  -Wstrict-aliasing=2     \
  -Wconversion            \
  -Wdouble-promotion      \
  -Wformat-security       \
  -Wimplicit-fallthrough  \
  -fno-omit-frame-pointer \
  -fstrict-aliasing       \
  -g                      \
  -O3

all: data_gen test

ALL_HEADERS=common.h othello.h wthor.h unit.h common.h

data_gen: Makefile data_gen.c common.c othello.c ${ALL_HEADERS}
	gcc ${CFLAGS} -o data_gen data_gen.c common.c othello.c

test: Makefile test_othello.c othello.c unit.c common.c ${ALL_HEADERS}
	gcc ${CFLAGS} -o test test_othello.c othello.c unit.c common.c
