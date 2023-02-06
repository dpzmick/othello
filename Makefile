flags=-Wall -Wextra

# ---------
# local system testing
test: test.c bitboard.h unit.h Makefile
	gcc test.c -o test ${flags}

run_%: %
	./$^

# ----------
# hacked out of the playdate's makefile
SDK_INCLUDE=-I/Users/dpzmick/Developer/PlaydateSDK/C_API/
SDK_SETUP=/Users/dpzmick/Developer/PlaydateSDK/C_API/buildsupport/setup.c

build/pdx.dylib: main.c bitboard.h
	mkdir -p build
	clang -dynamiclib -rdynamic -lm -DTARGET_SIMULATOR=1 -DTARGET_EXTENSION=1 $(SDK_INCLUDE) -o build/pdex.dylib main.c ${SDK_SETUP} ${flags}

simulate: build/pdx.dylib
	touch Source/pdex.bin
	cp build/pdex.dylib Source
	pdc Source Othello.pdx
	open Othello.pdx
