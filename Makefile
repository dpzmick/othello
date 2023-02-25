flags=-Wall -Wextra -Wconversion -Werror

# FIXME need to cross compile this!
xxhash_cflags=$(shell pkg-config --cflags libxxhash)
xxhash_ldflags=$(shell pkg-config --libs libxxhash)

# ---------
# local system testing
test: test.c unit.c test_board.c bitboard.h unit.h Makefile
	gcc ${xxhash_cflags} ${xxhash_ldflags} test.c unit.c test_board.c -o test ${flags}

run_%: %
	./$^

# ----------
# hacked out of the playdate's makefile
SDK_INCLUDE=-I/Users/dpzmick/Developer/PlaydateSDK/C_API/
SDK_SETUP=/Users/dpzmick/Developer/PlaydateSDK/C_API/buildsupport/setup.c

CROSS_CC=/usr/local/bin/arm-none-eabi-gcc
CROSS_OBJCOPY=/usr/local/bin/arm-none-eabi-objcopy
CROSS_CC_FLAGS=-mthumb -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -D__FPU_USED=1 \
	-O2 -falign-functions=16 -fomit-frame-pointer -gdwarf-2 \
	-Wall -Wno-unused -Wstrict-prototypes -Wno-unknown-pragmas \
	-fverbose-asm -Wdouble-promotion -ffunction-sections -fdata-sections \
	-Wa,-ahlms=build/main.lst -DTARGET_PLAYDATE=1 -DTARGET_EXTENSION=1

CROSS_LD_FLAGS=-mthumb -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -D__FPU_USED=1 -T/Users/dpzmick/Developer/PlaydateSDK/C_API/buildsupport/link_map.ld -Wl,-Map=build/pdex.map,--cref,--gc-sections,--no-warn-mismatch

build/pdx.dylib: main.c bitboard.h
	mkdir -p build
	clang -dynamiclib -rdynamic -lm -DTARGET_SIMULATOR=1 -DTARGET_EXTENSION=1 $(SDK_INCLUDE) ${flags} -o build/pdex.dylib main.c ${SDK_SETUP}

build/pdex.bin: main.c bitboard.h
	mkdir -p build
	/usr/local/bin/arm-none-eabi-gcc -g -c -mthumb -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -D__FPU_USED=1 -O2 -falign-functions=16 -fomit-frame-pointer -gdwarf-2 -Wall -Wno-unused -Wstrict-prototypes -Wno-unknown-pragmas -fverbose-asm -Wdouble-promotion -ffunction-sections -fdata-sections -Wa,-ahlms=build/main.lst -DTARGET_PLAYDATE=1 -DTARGET_EXTENSION=1  -I . -I . -I /Users/dpzmick/Developer/PlaydateSDK/C_API main.c -o build/main.o
	/usr/local/bin/arm-none-eabi-gcc -g -c -mthumb -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -D__FPU_USED=1 -O2 -falign-functions=16 -fomit-frame-pointer -gdwarf-2 -Wall -Wno-unused -Wstrict-prototypes -Wno-unknown-pragmas -fverbose-asm -Wdouble-promotion -ffunction-sections -fdata-sections -Wa,-ahlms=build/setup.lst -DTARGET_PLAYDATE=1 -DTARGET_EXTENSION=1  -I . -I . -I /Users/dpzmick/Developer/PlaydateSDK/C_API /Users/dpzmick/Developer/PlaydateSDK/C_API/buildsupport/setup.c -o build/setup.o
	/usr/local/bin/arm-none-eabi-gcc -g build/main.o build/setup.o -mthumb -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -D__FPU_USED=1 -T/Users/dpzmick/Developer/PlaydateSDK/C_API/buildsupport/link_map.ld -Wl,-Map=build/pdex.map,--cref,--gc-sections,--no-warn-mismatch    -o build/pdex.elf
	/usr/local/bin/arm-none-eabi-objcopy -O binary build/pdex.elf build/pdex.bin

simulate: build/pdx.dylib build/pdex.bin
	cp build/pdex.dylib Source
	cp build/pdex.bin Source
	pdc Source Othello.pdx
	open Othello.pdx
