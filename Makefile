flags=-Wall -Wextra

test: test.c bitboard.c bitboard.h unit.h Makefile
	gcc test.c bitboard.c -o test ${flags}

run_%: %
	./$^
